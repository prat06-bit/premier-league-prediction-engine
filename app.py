import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="KICKIQ · EPL Predictor", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ── GUARANTEED PADDING KILLER ─────────────────────────────────────────────────
# components.v1.html runs in real document context (not sandboxed)
components.html("""<script>
(function(){
  function nuke(){
    ['header[data-testid="stHeader"]','[data-testid="stDecoration"]','[data-testid="stToolbar"]'].forEach(function(s){
      var el=document.querySelector(s); if(el){el.style.setProperty('display','none','important');el.style.setProperty('height','0','important');}
    });
    ['.block-container','.stMainBlockContainer','[data-testid="stAppViewBlockContainer"]',
     '[data-testid="stVerticalBlock"]','section.main > div'].forEach(function(s){
      document.querySelectorAll(s).forEach(function(el){
        el.style.setProperty('padding-top','0','important');
        el.style.setProperty('padding-bottom','0','important');
        el.style.setProperty('padding-left','0','important');
        el.style.setProperty('padding-right','0','important');
        el.style.setProperty('margin-top','0','important');
        el.style.setProperty('margin-left','0','important');
        el.style.setProperty('margin-right','0','important');
        el.style.setProperty('gap','0','important');
        el.style.setProperty('max-width','100%','important');
        el.style.setProperty('width','100%','important');
      });
    });
    // No JS back-button override needed — using HTML overlay approach
  }
  nuke();
  new MutationObserver(nuke).observe(document.documentElement,{childList:true,subtree:true});
  var n=0,id=setInterval(function(){nuke();if(++n>80)clearInterval(id);},80);
})();
</script>""", height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=DM+Mono:wght@300;400;500&family=Rajdhani:wght@300;400;500;600;700&family=Saira+Condensed:wght@200;300;400;500;600;700;800;900&display=swap');
*,*::before,*::after{box-sizing:border-box;}
:root{--acid:#C8FF00;--void:#050608;--surface:#0A0C0F;--surface2:#0F1115;--border:rgba(255,255,255,0.06);--text:#F4F4F5;--muted:rgba(244,244,245,0.4);--muted2:rgba(244,244,245,0.22);--red:#FF3A3A;--amber:#FFB800;--cyan:#00E5FF;}
html,body,.stApp{background:var(--void) !important;color:var(--text);font-family:'Rajdhani',sans-serif;overflow-x:hidden;}
header[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stToolbar"],#MainMenu,footer{display:none !important;height:0 !important;}
.block-container,.stMainBlockContainer,[data-testid="stAppViewBlockContainer"],section.main>div,div.appview-container section.main>div:first-child{padding-top:0 !important;padding-bottom:0 !important;padding-left:0 !important;padding-right:0 !important;margin-top:0 !important;margin-left:0 !important;margin-right:0 !important;gap:0 !important;max-width:100% !important;width:100% !important;}
[data-testid="stVerticalBlock"]{gap:0 !important;padding:0 !important;margin:0 !important;}
iframe{display:none !important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:var(--void);}::-webkit-scrollbar-thumb{background:var(--acid);border-radius:2px;}
body::after{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.028'/%3E%3C/svg%3E");pointer-events:none;z-index:9999;mix-blend-mode:overlay;}
.orb-container{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(90px);animation:orbFloat var(--dur,22s) ease-in-out infinite;}
.orb-1{width:600px;height:600px;top:-15%;left:-10%;background:radial-gradient(circle,rgba(200,255,0,0.1) 0%,transparent 70%);--dur:20s;}
.orb-2{width:450px;height:450px;bottom:-10%;right:-8%;background:radial-gradient(circle,rgba(0,229,255,0.07) 0%,transparent 70%);--dur:25s;animation-delay:-8s;animation-direction:reverse;}
.orb-3{width:350px;height:350px;top:50%;left:55%;background:radial-gradient(circle,rgba(200,255,0,0.04) 0%,transparent 70%);--dur:32s;animation-delay:-14s;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}33%{transform:translate(25px,-35px) scale(1.04);}66%{transform:translate(-18px,28px) scale(0.96);}}
.bg-grid{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;background-image:linear-gradient(rgba(200,255,0,0.022) 1px,transparent 1px),linear-gradient(90deg,rgba(200,255,0,0.022) 1px,transparent 1px);background-size:80px 80px;mask-image:radial-gradient(ellipse 75% 75% at 50% 50%,black 25%,transparent 100%);}

/* ═══ LANDING ═══ */
.landing-wrap{width:100vw;max-width:100%;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:5vh 2rem 0;position:relative;z-index:1;box-sizing:border-box;}
.landing-wrap > *{width:100%;max-width:860px;margin-left:auto;margin-right:auto;}
.eyebrow{display:inline-flex;align-items:center;gap:10px;background:linear-gradient(135deg,rgba(200,255,0,0.08),rgba(200,255,0,0.02));border:1px solid rgba(200,255,0,0.2);border-radius:100px;padding:6px 18px 6px 10px;font-family:'DM Mono',monospace;font-size:0.66rem;letter-spacing:0.18em;color:var(--acid);text-transform:uppercase;margin-bottom:2rem;animation:fadeDown 0.6s cubic-bezier(0.16,1,0.3,1) 0.05s both;}
.eyebrow-pulse{width:7px;height:7px;background:var(--acid);border-radius:50%;animation:pls 1.8s ease-in-out infinite;box-shadow:0 0 8px var(--acid);}
@keyframes pls{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.3;transform:scale(0.65);}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-12px);}to{opacity:1;transform:translateY(0);}}
.hero-wordmark{font-family:'Anton',sans-serif;font-size:clamp(5.5rem,16vw,13rem);line-height:0.85;letter-spacing:-0.02em;text-align:center;position:relative;z-index:1;animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.1s both;}
.wm-kick{color:var(--text);}
.wm-iq{color:var(--acid);text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);animation:acidFlicker 6s ease-in-out infinite 2.5s;}
@keyframes heroUp{from{opacity:0;transform:translateY(28px) skewY(1.5deg);}to{opacity:1;transform:translateY(0) skewY(0);}}
@keyframes acidFlicker{0%,88%,100%{text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);}90%{text-shadow:0 0 6px rgba(200,255,0,0.2);opacity:0.85;}92%{text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);}95%{text-shadow:0 0 6px rgba(200,255,0,0.2);opacity:0.78;}}
.hero-tagline{font-family:'Saira Condensed',sans-serif;font-weight:300;font-size:clamp(0.9rem,2vw,1.1rem);letter-spacing:0.42em;text-transform:uppercase;color:var(--muted);text-align:center;margin:1.4rem 0 0;animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.22s both;}
.metrics-strip{display:flex;gap:0;align-items:stretch;margin:3.5rem auto 0;width:100%;max-width:700px;border:1px solid var(--border);border-radius:16px;overflow:hidden;background:var(--surface);animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.38s both;}
.metric-block{flex:1;padding:1.8rem 2rem;position:relative;border-right:1px solid var(--border);text-align:center;transition:background 0.3s;}
.metric-block:last-child{border-right:none;}.metric-block:hover{background:rgba(200,255,0,0.03);}
.metric-block::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--acid),transparent);transform:scaleX(0);transition:transform 0.35s;}
.metric-block:hover::after{transform:scaleX(1);}
.metric-val{font-family:'Anton',sans-serif;font-size:2.8rem;line-height:1;color:var(--acid);animation:metricPop 0.7s cubic-bezier(0.34,1.56,0.64,1) both;}
.metric-block:nth-child(1) .metric-val{animation-delay:0.55s;}.metric-block:nth-child(2) .metric-val{animation-delay:0.67s;}.metric-block:nth-child(3) .metric-val{animation-delay:0.79s;}.metric-block:nth-child(4) .metric-val{animation-delay:0.91s;}
@keyframes metricPop{0%{opacity:0;transform:scale(0.55);}65%{transform:scale(1.12);opacity:1;}100%{transform:scale(1);}}
.metric-lbl{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin-top:5px;}
/* ── ALL buttons default: big green acid style ── */
.stButton>button{
  background:var(--acid) !important;color:#050608 !important;border:none !important;
  border-radius:12px !important;font-family:'Anton',sans-serif !important;
  font-size:1.25rem !important;letter-spacing:0.1em !important;
  padding:1rem 2.5rem !important;width:100% !important;
  transition:all 0.22s cubic-bezier(0.34,1.56,0.64,1) !important;
  animation:ctaPulse 4s ease-in-out infinite !important;
}
.stButton>button:hover{transform:translateY(-3px) scale(1.02) !important;box-shadow:0 0 50px rgba(200,255,0,0.6),0 8px 30px rgba(200,255,0,0.25) !important;animation:none !important;}
.stButton>button:active{transform:scale(0.98) !important;}
@keyframes ctaPulse{0%,100%{box-shadow:0 0 0 rgba(200,255,0,0);}50%{box-shadow:0 0 38px rgba(200,255,0,0.38);}}

/* ── BACK button override — targets by key using data-testid ── */
/* Streamlit renders: <div data-testid="stButton"><button>← BACK</button></div> */
/* We target the button whose text starts with ← via the parent having key=back_btn */
.cta-wrap{margin-top:2.5rem;position:relative;z-index:1;}
.sec-header{text-align:center;margin:7rem 0 3.5rem;position:relative;z-index:1;}
.sec-label{font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.32em;color:var(--acid);text-transform:uppercase;margin-bottom:0.8rem;}
.sec-title{font-family:'Anton',sans-serif;font-size:clamp(2.5rem,5vw,4rem);letter-spacing:0.01em;color:var(--text);line-height:1;}
.sec-sub{font-family:'Saira Condensed',sans-serif;font-weight:300;font-size:1rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase;margin-top:0.8rem;}
.hiw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;width:100%;max-width:900px;margin:0 auto;position:relative;z-index:1;}
.hiw-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:2rem 1.8rem;position:relative;overflow:hidden;transition:transform 0.3s,border-color 0.3s,box-shadow 0.3s;opacity:0;transform:translateY(30px);animation:revealUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards;}
.hiw-card:nth-child(1){animation-delay:0.1s;}.hiw-card:nth-child(2){animation-delay:0.22s;}.hiw-card:nth-child(3){animation-delay:0.34s;}
.hiw-card:hover{transform:translateY(-5px);border-color:rgba(200,255,0,0.2);box-shadow:0 12px 50px rgba(0,0,0,0.5),0 0 25px rgba(200,255,0,0.06);}
.hiw-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(200,255,0,0.3),transparent);opacity:0;transition:opacity 0.3s;}
.hiw-card:hover::before{opacity:1;}
.hiw-num{font-family:'Anton',sans-serif;font-size:4rem;line-height:1;color:rgba(200,255,0,0.07);margin-bottom:1rem;}
.hiw-icon{font-size:1.8rem;margin-bottom:1rem;}.hiw-title{font-family:'Anton',sans-serif;font-size:1.3rem;letter-spacing:0.04em;color:var(--text);margin-bottom:0.6rem;}
.hiw-desc{font-family:'Rajdhani',sans-serif;font-size:0.95rem;font-weight:400;color:var(--muted);line-height:1.6;}
@keyframes revealUp{from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);}}
.ticker-wrap{width:100%;overflow:hidden;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:rgba(200,255,0,0.02);padding:1rem 0;margin:7rem 0 0;position:relative;z-index:1;}
.ticker-inner{display:flex;gap:4rem;animation:ticker 32s linear infinite;white-space:nowrap;}
.ticker-item{display:flex;align-items:center;gap:1rem;flex-shrink:0;font-family:'DM Mono',monospace;font-size:0.66rem;letter-spacing:0.16em;color:var(--muted);text-transform:uppercase;}
.ticker-dot{width:4px;height:4px;background:var(--acid);border-radius:50%;flex-shrink:0;}
@keyframes ticker{from{transform:translateX(0);}to{transform:translateX(-50%);}}
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;width:100%;max-width:860px;margin:7rem auto 0;background:var(--border);border-radius:18px;overflow:hidden;position:relative;z-index:1;}
.stat-block{background:var(--surface);padding:2.5rem 1.5rem;text-align:center;transition:background 0.3s;position:relative;overflow:hidden;}
.stat-block:hover{background:rgba(200,255,0,0.03);}
.stat-block::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--acid),transparent);transform:scaleX(0);transition:transform 0.4s;}
.stat-block:hover::after{transform:scaleX(1);}
.stat-num{font-family:'Anton',sans-serif;font-size:2.8rem;line-height:1;color:var(--acid);}
.stat-unit{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;color:var(--muted);text-transform:uppercase;margin-top:5px;}
.stat-desc{font-family:'Rajdhani',sans-serif;font-size:0.9rem;font-weight:400;color:var(--muted2);margin-top:8px;line-height:1.5;}
.vs-section{width:100%;max-width:860px;margin:7rem auto 0;position:relative;z-index:1;}
.compare-grid{display:grid;grid-template-columns:1fr 44px 1fr;gap:0;border-radius:18px;overflow:hidden;border:1px solid var(--border);}
.compare-col{background:var(--surface);padding:2rem;}
.compare-col.ours{background:linear-gradient(140deg,rgba(200,255,0,0.07) 0%,rgba(200,255,0,0.02) 100%);border-right:1px solid rgba(200,255,0,0.12);}
.compare-col-label{font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:1.4rem;color:var(--muted);}
.compare-col.ours .compare-col-label{color:var(--acid);}
.compare-row{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.03);font-family:'Rajdhani',sans-serif;font-size:0.95rem;font-weight:500;color:rgba(244,244,245,0.62);opacity:0;animation:rowSlide 0.4s ease forwards;}
.compare-col.ours .compare-row{color:rgba(244,244,245,0.82);}
.compare-row:last-child{border-bottom:none;}
.compare-row:nth-child(2){animation-delay:0.8s;}.compare-row:nth-child(3){animation-delay:0.9s;}.compare-row:nth-child(4){animation-delay:1.0s;}.compare-row:nth-child(5){animation-delay:1.1s;}.compare-row:nth-child(6){animation-delay:1.2s;}.compare-row:nth-child(7){animation-delay:1.3s;}.compare-row:nth-child(8){animation-delay:1.4s;}.compare-row:nth-child(9){animation-delay:1.5s;}.compare-row:nth-child(10){animation-delay:1.6s;}
@keyframes rowSlide{from{opacity:0;transform:translateX(-8px);}to{opacity:1;transform:translateX(0);}}
.c-check{color:var(--acid);font-size:0.82rem;}.c-cross{color:rgba(244,244,245,0.16);font-size:0.82rem;}
.compare-divider{background:var(--surface2);display:flex;align-items:center;justify-content:center;border-left:1px solid rgba(200,255,0,0.08);border-right:1px solid rgba(200,255,0,0.08);}
.vs-pill{font-family:'Anton',sans-serif;font-size:0.95rem;color:rgba(255,255,255,0.08);writing-mode:vertical-rl;letter-spacing:0.5em;}
.landing-footer{font-family:'DM Mono',monospace;font-size:0.56rem;letter-spacing:0.12em;color:rgba(244,244,245,0.16);text-align:center;text-transform:uppercase;margin-top:6rem;padding-bottom:4rem;line-height:2.4;position:relative;z-index:1;}
.footer-line{height:1px;background:linear-gradient(90deg,transparent,rgba(200,255,0,0.15),transparent);margin:3rem auto 2.5rem;max-width:600px;}
/* ── Feature grid ── */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;width:100%;max-width:900px;margin:0 auto;position:relative;z-index:1;}
.feat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.6rem 1.5rem;transition:border-color 0.3s,transform 0.3s;animation:revealUp 0.5s cubic-bezier(0.16,1,0.3,1) both;}
.feat-card:nth-child(1){animation-delay:0.05s;}.feat-card:nth-child(2){animation-delay:0.12s;}.feat-card:nth-child(3){animation-delay:0.19s;}.feat-card:nth-child(4){animation-delay:0.26s;}.feat-card:nth-child(5){animation-delay:0.33s;}.feat-card:nth-child(6){animation-delay:0.40s;}
.feat-card:hover{border-color:rgba(200,255,0,0.18);transform:translateY(-3px);}
.feat-icon{font-size:1.6rem;margin-bottom:0.8rem;}
.feat-title{font-family:'Anton',sans-serif;font-size:1rem;letter-spacing:0.05em;color:var(--text);margin-bottom:0.5rem;}
.feat-desc{font-family:'Rajdhani',sans-serif;font-size:0.88rem;font-weight:400;color:var(--muted);line-height:1.55;}
/* ── FAQ ── */
.faq-list{width:100%;max-width:760px;margin:0 auto;position:relative;z-index:1;}
.faq-item{border-bottom:1px solid rgba(255,255,255,0.05);padding:1.6rem 0;animation:revealUp 0.5s cubic-bezier(0.16,1,0.3,1) both;}
.faq-item:nth-child(1){animation-delay:0.05s;}.faq-item:nth-child(2){animation-delay:0.15s;}.faq-item:nth-child(3){animation-delay:0.25s;}.faq-item:nth-child(4){animation-delay:0.35s;}
.faq-item:last-child{border-bottom:none;}
.faq-q{font-family:'Anton',sans-serif;font-size:1.1rem;letter-spacing:0.03em;color:var(--text);margin-bottom:0.6rem;display:flex;align-items:center;gap:10px;}
.faq-q::before{content:'';width:4px;height:1.1rem;background:var(--acid);border-radius:2px;flex-shrink:0;}
.faq-a{font-family:'Rajdhani',sans-serif;font-size:0.95rem;font-weight:400;color:var(--muted);line-height:1.65;padding-left:14px;}

/* ═══ PREDICT PAGE ═══ */
.pred-topbar{display:flex;align-items:center;justify-content:space-between;padding:1.2rem 3rem;border-bottom:1px solid rgba(200,255,0,0.07);position:sticky;top:0;z-index:200;background:rgba(5,6,8,0.72);backdrop-filter:blur(22px) saturate(160%);width:100%;animation:topbarDrop 0.5s cubic-bezier(0.16,1,0.3,1) both;}
@keyframes topbarDrop{from{opacity:0;transform:translateY(-100%);}to{opacity:1;transform:translateY(0);}}
.pred-logo-text{font-family:'Anton',sans-serif;font-size:1.65rem;letter-spacing:0.04em;color:var(--text);line-height:1;}
.pred-logo-text .iq{color:var(--acid);}
.pred-subtitle{font-family:'DM Mono',monospace;font-size:0.56rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase;margin-top:3px;}
.live-dot{width:5px;height:5px;background:var(--acid);border-radius:50%;display:inline-block;box-shadow:0 0 8px var(--acid);animation:pls 1.5s ease-in-out infinite;margin-right:6px;}
.topbar-tag{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.14em;color:rgba(244,244,245,0.2);text-transform:uppercase;}
/* Selector zone */
.selector-zone{padding:3rem 2rem 1.5rem;width:100%;animation:heroUp 0.6s cubic-bezier(0.16,1,0.3,1) 0.1s both;}
.stSelectbox>div>div{background:var(--surface) !important;border:1px solid rgba(255,255,255,0.09) !important;border-radius:10px !important;color:var(--text) !important;font-family:'Rajdhani',sans-serif !important;font-size:1.05rem !important;font-weight:600 !important;transition:all .2s ease !important;}
.stSelectbox>div>div:hover{border-color:rgba(200,255,0,0.38) !important;}
.stSelectbox:focus-within div[data-baseweb="select"]{box-shadow:0 0 0 1px rgba(200,255,0,0.5),0 0 18px rgba(200,255,0,0.12) !important;border-color:rgba(200,255,0,0.45) !important;}
.stSelectbox label{font-family:'DM Mono',monospace !important;font-size:0.6rem !important;letter-spacing:0.16em !important;text-transform:uppercase !important;color:var(--muted) !important;}
.result-hero{background:linear-gradient(135deg,rgba(200,255,0,0.07),rgba(200,255,0,0.02));border:1px solid rgba(200,255,0,0.2);border-radius:20px;padding:3rem;text-align:center;position:relative;overflow:hidden;margin:1.5rem 0;animation:cardUp 0.6s cubic-bezier(0.16,1,0.3,1) both;transition:box-shadow 0.4s;}
.result-hero:hover{box-shadow:0 0 70px rgba(200,255,0,0.2);}
.result-hero::before{content:'';position:absolute;top:0;left:-200%;width:200%;height:1px;background:linear-gradient(90deg,transparent,var(--acid),transparent);animation:scanline 3s linear infinite;}
.result-hero::after{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 60% 60% at 50% 0%,rgba(200,255,0,0.08),transparent 70%);pointer-events:none;}
@keyframes scanline{from{left:-200%;}to{left:200%;}}
@keyframes cardUp{from{opacity:0;transform:translateY(35px);}to{opacity:1;transform:translateY(0);}}
.result-eyebrow{font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;color:var(--muted);text-transform:uppercase;margin-bottom:1rem;}
.result-outcome-text{font-family:'Anton',sans-serif;font-size:clamp(3.5rem,8vw,5.5rem);line-height:1;color:var(--acid);text-shadow:0 0 60px rgba(200,255,0,0.4);letter-spacing:0.02em;position:relative;z-index:1;overflow:hidden;animation:cinematicReveal 0.7s cubic-bezier(0.16,1,0.3,1) 0.1s both;}
.result-outcome-text::after{content:'';position:absolute;top:0;left:-130%;width:130%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);animation:broadcastSweep 1.2s ease 0.3s forwards;}
@keyframes broadcastSweep{to{left:130%;}}
@keyframes cinematicReveal{0%{opacity:0;transform:translateY(40px) scale(0.92);filter:blur(8px);}60%{opacity:1;filter:blur(0);}100%{transform:translateY(0) scale(1);}}
.result-matchup-text{font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase;margin-top:0.5rem;position:relative;z-index:1;}
.signal-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 18px;border-radius:100px;font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.12em;font-weight:500;margin:1.2rem 0 0.4rem;position:relative;z-index:1;animation:badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.55s both;}
@keyframes badgePop{from{opacity:0;transform:scale(0.6);}to{opacity:1;transform:scale(1);}}
.signal-badge.s-strong{background:rgba(200,255,0,0.1);border:1px solid rgba(200,255,0,0.35);color:var(--acid);}
.signal-badge.s-mod{background:rgba(255,184,0,0.1);border:1px solid rgba(255,184,0,0.3);color:var(--amber);}
.signal-badge.s-weak{background:rgba(255,100,0,0.1);border:1px solid rgba(255,100,0,0.3);color:#FF6400;}
.signal-badge.s-none{background:rgba(255,58,58,0.1);border:1px solid rgba(255,58,58,0.3);color:var(--red);}
.disclaimer-txt{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;color:rgba(244,244,245,0.18);text-transform:uppercase;margin-top:8px;position:relative;z-index:1;}
.prob-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0;}
.prob-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.8rem 1.5rem;text-align:center;transition:all 0.3s cubic-bezier(0.34,1.56,0.64,1);position:relative;overflow:hidden;opacity:0;animation:cardUp 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;}
.prob-card:nth-child(1){animation-delay:0.2s;}.prob-card:nth-child(2){animation-delay:0.35s;}.prob-card:nth-child(3){animation-delay:0.5s;}
.prob-card:hover{transform:translateY(-5px) scale(1.02);border-color:rgba(200,255,0,0.2);box-shadow:0 0 28px rgba(200,255,0,0.1);}
.prob-card.active{background:linear-gradient(135deg,rgba(200,255,0,0.1),rgba(200,255,0,0.04));border-color:rgba(200,255,0,0.38);box-shadow:0 0 35px rgba(200,255,0,0.1);}
.prob-card.active::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--acid),transparent);}
.prob-pct{font-family:'Anton',sans-serif;font-size:3rem;line-height:1;color:var(--text);transition:color 0.3s;animation:numPop 0.6s cubic-bezier(0.34,1.56,0.64,1) both;}
.prob-card.active .prob-pct{color:var(--acid);}
.prob-label{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin-top:6px;}
@keyframes numPop{0%{transform:scale(0.5);opacity:0;}70%{transform:scale(1.12);opacity:1;}100%{transform:scale(1);}}
.intel-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;position:relative;overflow:hidden;transition:border-color 0.3s,transform 0.3s,box-shadow 0.3s;animation:cardUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.3s both;}
.intel-card:hover{border-color:rgba(200,255,0,0.16);transform:translateY(-3px);box-shadow:0 10px 45px rgba(0,0,0,0.45);}
.intel-card-kicker{font-family:'DM Mono',monospace;font-size:0.57rem;letter-spacing:0.2em;color:var(--acid);text-transform:uppercase;margin-bottom:1.3rem;display:flex;align-items:center;gap:8px;}
.intel-metrics{display:flex;gap:0.6rem;margin-bottom:0.6rem;flex-wrap:wrap;}
.intel-metric{flex:1;min-width:65px;background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:0.9rem 0.7rem;transition:border-color 0.2s,background 0.2s;}
.intel-metric:hover{border-color:rgba(200,255,0,0.14);background:rgba(200,255,0,0.02);}
.intel-metric-val{font-family:'Anton',sans-serif;font-size:1.9rem;line-height:1;color:var(--text);}
.intel-metric-lbl{font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:0.1em;color:var(--muted);text-transform:uppercase;margin-top:4px;}
.form-row{display:flex;gap:4px;flex-wrap:wrap;margin-top:0.5rem;}
.fp{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:7px;font-family:'DM Mono',monospace;font-size:0.7rem;font-weight:500;position:relative;overflow:hidden;opacity:0;animation:pillPop 0.32s cubic-bezier(0.34,1.56,0.64,1) forwards;}
.fp:nth-child(1){animation-delay:0.38s;}.fp:nth-child(2){animation-delay:0.50s;}.fp:nth-child(3){animation-delay:0.62s;}.fp:nth-child(4){animation-delay:0.74s;}.fp:nth-child(5){animation-delay:0.86s;}
@keyframes pillPop{from{opacity:0;transform:scale(0.25) rotate(-25deg);}to{opacity:1;transform:scale(1) rotate(0);}}
.fp::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,0.1),transparent);}
.fp-W{background:rgba(200,255,0,0.18);color:var(--acid);border:1px solid rgba(200,255,0,0.28);}
.fp-D{background:rgba(255,184,0,0.15);color:var(--amber);border:1px solid rgba(255,184,0,0.22);}
.fp-L{background:rgba(255,58,58,0.12);color:var(--red);border:1px solid rgba(255,58,58,0.18);}
.h2h-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;margin:1.5rem 0;animation:cardUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.4s both;}
.h2h-stat{font-family:'Rajdhani',sans-serif;font-weight:500;font-size:0.95rem;color:rgba(244,244,245,0.58);margin-top:1rem;line-height:1.6;display:flex;align-items:flex-start;gap:10px;}
.notes-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;margin:1.5rem 0;animation:cardUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.5s both;}
.notes-item{display:flex;gap:12px;padding:11px 0;border-bottom:1px solid rgba(255,255,255,0.025);font-family:'Rajdhani',sans-serif;font-weight:500;font-size:0.95rem;color:rgba(244,244,245,0.7);line-height:1.55;opacity:0;animation:rowSlide 0.38s ease forwards;}
.notes-item:nth-child(2){animation-delay:0.55s;}.notes-item:nth-child(3){animation-delay:0.65s;}.notes-item:nth-child(4){animation-delay:0.75s;}.notes-item:nth-child(5){animation-delay:0.85s;}.notes-item:nth-child(6){animation-delay:0.95s;}.notes-item:nth-child(7){animation-delay:1.05s;}.notes-item:nth-child(8){animation-delay:1.15s;}.notes-item:nth-child(9){animation-delay:1.25s;}.notes-item:nth-child(10){animation-delay:1.35s;}
.notes-item:last-child{border-bottom:none;}.note-icon{flex-shrink:0;font-size:1rem;margin-top:1px;}
.extra-insight{background:linear-gradient(135deg,rgba(0,229,255,0.05),rgba(0,229,255,0.01));border:1px solid rgba(0,229,255,0.14);border-radius:16px;padding:1.8rem 2rem;margin:1rem 0;animation:cardUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.6s both;}
.extra-insight .intel-card-kicker{color:var(--cyan);}
.insight-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;}
.insight-item{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1rem;}
.insight-item-label{font-family:'DM Mono',monospace;font-size:0.54rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin-bottom:5px;}
.insight-item-val{font-family:'Anton',sans-serif;font-size:1.4rem;color:var(--text);line-height:1;}
.insight-item-val.good{color:var(--acid);}.insight-item-val.bad{color:var(--red);}
.section-tag{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;color:var(--acid);text-transform:uppercase;margin-bottom:1rem;display:flex;align-items:center;gap:10px;}
.section-tag::before{content:'';width:18px;height:1px;background:var(--acid);flex-shrink:0;}
.line-sep{height:1px;background:linear-gradient(90deg,transparent,rgba(200,255,0,0.28),transparent);margin:2.5rem 0;animation:linePulse 3s ease-in-out infinite;}
@keyframes linePulse{0%,100%{opacity:0.32;}50%{opacity:1;}}
.streamlit-expanderHeader{font-family:'DM Mono',monospace !important;font-size:0.66rem !important;letter-spacing:0.15em !important;color:var(--muted) !important;background:var(--surface) !important;border:1px solid var(--border) !important;border-radius:10px !important;}
.streamlit-expanderContent{background:var(--surface) !important;border:1px solid var(--border) !important;border-top:none !important;}
.stSpinner>div{border-top-color:var(--acid) !important;}
@media(max-width:768px){.hero-wordmark{font-size:4.5rem;}.metrics-strip{flex-direction:column;}.hiw-grid{grid-template-columns:1fr;}.stats-row{grid-template-columns:1fr 1fr;}.compare-grid{grid-template-columns:1fr;}.compare-divider{display:none;}.selector-zone{padding:2rem 1.5rem;}.pred-topbar{padding:1.2rem 1.5rem;}.prob-grid{grid-template-columns:1fr 1fr;}.prob-grid .prob-card:last-child{grid-column:1/-1;}.insight-row{grid-template-columns:1fr;}.landing-wrap{padding-top:4vh;}}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        xgb = joblib.load('models/tuned/xgboost_tuned.pkl')
        rf  = joblib.load('models/tuned/random_forest_tuned.pkl')
        fc  = joblib.load('models/tuned/feature_columns.pkl')
        return xgb, rf, fc, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, False

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/features.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        return df, teams, True
    except:
        return None, None, False

def get_team_form(df, team, n=5):
    home = df[df['HomeTeam']==team][['Date','FTR']].copy()
    home['result'] = home['FTR'].map({'H':'W','D':'D','A':'L'})
    away = df[df['AwayTeam']==team][['Date','FTR']].copy()
    away['result'] = away['FTR'].map({'H':'L','D':'D','A':'W'})
    combined = pd.concat([home[['Date','result']], away[['Date','result']]])
    return combined.sort_values('Date', ascending=False).head(n)['result'].tolist()

def get_team_stats(df, team):
    home = df[df['HomeTeam']==team].copy()
    away = df[df['AwayTeam']==team].copy()
    all_m = pd.concat([
        home[['Date','FTHG','FTAG','FTR']].assign(venue='home'),
        away[['Date','FTHG','FTAG','FTR']].assign(venue='away')
    ]).sort_values('Date', ascending=False).head(38)
    if len(all_m)==0:
        return {'played':0,'wins':0,'draws':0,'losses':0,'goals_for':0,'goals_against':0,'gd':0,'win_rate':0,'clean_sheets':0,'btts':0}
    wins   = (((all_m['venue']=='home')&(all_m['FTR']=='H'))|((all_m['venue']=='away')&(all_m['FTR']=='A'))).sum()
    draws  = (all_m['FTR']=='D').sum()
    losses = len(all_m)-wins-draws
    gf = int(np.where(all_m['venue']=='home', all_m['FTHG'], all_m['FTAG']).sum())
    ga = int(np.where(all_m['venue']=='home', all_m['FTAG'], all_m['FTHG']).sum())
    clean_sheets = int((np.where(all_m['venue']=='home', all_m['FTAG'], all_m['FTHG'])==0).sum())
    btts = int(((all_m['FTHG']>0)&(all_m['FTAG']>0)).sum())
    return {'played':len(all_m),'wins':int(wins),'draws':int(draws),'losses':int(losses),
            'goals_for':gf,'goals_against':ga,'gd':gf-ga,'win_rate':round(wins/len(all_m)*100),
            'clean_sheets':clean_sheets,'btts':btts}

def get_h2h(df, home, away, n=5):
    h2h = df[((df['HomeTeam']==home)&(df['AwayTeam']==away))|
             ((df['HomeTeam']==away)&(df['AwayTeam']==home))]
    h2h = h2h.sort_values('Date', ascending=False).head(n)
    results = []
    for _, row in h2h.iterrows():
        if row['HomeTeam']==home:
            results.append({'H':'W','D':'D','A':'L'}[row['FTR']])
        else:
            results.append({'A':'W','D':'D','H':'L'}[row['FTR']])
    return results

def run_prediction(home, away, xgb, rf, df, fc):
    hr = df[df['HomeTeam']==home].sort_values('Date', ascending=False)
    ar = df[df['AwayTeam']==away].sort_values('Date', ascending=False)
    if len(hr)==0 or len(ar)==0: return None
    home_row, away_row = hr.iloc[0], ar.iloc[0]
    feat_dict = {}
    for col in fc:
        if '_home' in col: feat_dict[col] = home_row.get(col, 0)
        elif '_away' in col: feat_dict[col] = away_row.get(col, 0)
        else: feat_dict[col] = 0
    feats = pd.DataFrame([feat_dict])[fc].fillna(0)
    xp = xgb.predict_proba(feats)[0]; rp = rf.predict_proba(feats)[0]
    ens = 0.6*xp + 0.4*rp
    return {'proba':ens,'outcome':['Home Win','Draw','Away Win'][np.argmax(ens)],'confidence':ens.max(),'xgb':xp,'rf':rp}

def form_html(form):
    return '<div class="form-row">'+''.join(f'<span class="fp fp-{r}">{r}</span>' for r in form)+'</div>'

def signal_badge_html(conf):
    if conf>=0.65: return '<div class="signal-badge s-strong">⚡ STRONG SIGNAL</div>'
    elif conf>=0.55: return '<div class="signal-badge s-mod">◈ MODERATE SIGNAL</div>'
    elif conf>=0.45: return '<div class="signal-badge s-weak">⚠ WEAK SIGNAL</div>'
    else: return '<div class="signal-badge s-none">✕ NO CLEAR EDGE</div>'

def insider_notes(home, away, res, hs, as_, h2h):
    notes = []
    idx = np.argmax(res['proba'])
    notes.append(("🤖", f"Ensemble: XGBoost ({res['xgb'][idx]*100:.0f}%) + Random Forest ({res['rf'][idx]*100:.0f}%) weighted 60/40."))
    if res['confidence']>0.60: notes.append(("🔥", f"High conviction — {res['outcome']} is the dominant call across both models."))
    else: notes.append(("🌊", "Probabilities are spread — contested fixture, upsets very possible."))
    gd_diff = hs['gd']-as_['gd']
    if abs(gd_diff)>8:
        better = home if gd_diff>0 else away
        notes.append(("📈", f"{better} hold significantly stronger goal difference — superior attacking & defensive quality."))
    if hs['win_rate']>60: notes.append(("🏠", f"{home} winning {hs['win_rate']}% of recent matches — elite home form."))
    if as_['win_rate']>60: notes.append(("✈️", f"{away} carry {as_['win_rate']}% win rate — dangerous regardless of venue."))
    if hs['clean_sheets']>=5: notes.append(("🧤", f"{home} kept {hs['clean_sheets']} clean sheets recently — rock-solid defensive unit."))
    if as_['clean_sheets']>=5: notes.append(("🛡️", f"{away} boast {as_['clean_sheets']} clean sheets — expect a tight game."))
    if h2h:
        hw=h2h.count('W'); hd=h2h.count('D'); hl=h2h.count('L')
        verdict = 'History favours the home side.' if hw>hl else 'Away side historically dominant.' if hl>hw else 'Historically level — anyone can win.'
        notes.append(("⚔️", f"H2H ({home}): {hw}W · {hd}D · {hl}L from last {len(h2h)} meetings. {verdict}"))
    if res['proba'][1]>0.30: notes.append(("🤝", f"Draw probability at {res['proba'][1]*100:.0f}% — cagey, tactical fixture expected."))
    if hs['btts']>=6 and as_['btts']>=6: notes.append(("⚽", f"Both teams to score is very likely — both sides have featured in high-scoring games recently."))
    notes.append(("⚠️", "Injuries, suspensions, weather & referee tendencies are NOT modelled. Always contextualise."))
    return notes

# ─── STATE + BACKGROUND ───────────────────────────────────────────────────────
if 'page'   not in st.session_state: st.session_state.page   = 'landing'
if 'result' not in st.session_state: st.session_state.result = None

st.markdown("""
<div class="orb-container"><div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div></div>
<div class="bg-grid"></div>
""", unsafe_allow_html=True)

xgb_model, rf_model, feature_cols, models_ok = load_models()
df, teams, data_ok = load_data()

# ═══════════════════════════════════════════════════════════
#  LANDING PAGE
# ═══════════════════════════════════════════════════════════
if st.session_state.page == 'landing':
    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;"><div class="eyebrow"><span class="eyebrow-pulse"></span>AI-POWERED · PREMIER LEAGUE · SEASON 2024/25</div></div>
    <div class="hero-wordmark"><span class="wm-kick">KICK</span><span class="wm-iq">IQ</span></div>
    <div class="hero-tagline">Predict the game — before the whistle blows</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-strip">
      <div class="metric-block"><div class="metric-val">3,800+</div><div class="metric-lbl">Matches Trained</div></div>
      <div class="metric-block"><div class="metric-val">219</div><div class="metric-lbl">Features Engineered</div></div>
      <div class="metric-block"><div class="metric-val">10Y</div><div class="metric-lbl">Historical Data</div></div>
      <div class="metric-block"><div class="metric-val">2</div><div class="metric-lbl">Stacked Models</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 1.4, 1])
    with col_c:
        if st.button("⚡  LAUNCH PREDICTION ENGINE", key="cta_btn"):
            st.session_state.page = 'predict'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    ticker_items = ["3,800+ matches analysed","XGBoost ensemble","10 seasons of EPL data","219 engineered features","Random Forest stacking","Real-time predictions","Head-to-head patterns","Form streak analysis","Goal difference trends","Confidence-scored signals","AI insider insights","Home/Away splits"]
    ticker_html = ''.join(f'<span class="ticker-item"><span class="ticker-dot"></span>{item}</span>' for item in ticker_items*2)
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-inner">{ticker_html}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-header">
      <div class="sec-label">THE PROCESS</div>
      <div class="sec-title">HOW IT WORKS</div>
      <div class="sec-sub">Three steps from selection to prediction</div>
    </div>
    <div class="hiw-grid">
      <div class="hiw-card"><div class="hiw-num">01</div><div class="hiw-icon">🎯</div><div class="hiw-title">SELECT MATCH</div><div class="hiw-desc">Choose any home and away team from the current EPL roster. KICKIQ covers all 20 Premier League clubs.</div></div>
      <div class="hiw-card"><div class="hiw-num">02</div><div class="hiw-icon">⚙️</div><div class="hiw-title">ENSEMBLE RUNS</div><div class="hiw-desc">XGBoost and Random Forest models process 219 features — form, goal difference, H2H history, home/away splits — weighted 60/40.</div></div>
      <div class="hiw-card"><div class="hiw-num">03</div><div class="hiw-icon">📊</div><div class="hiw-title">GET INSIGHTS</div><div class="hiw-desc">Receive outcome probabilities, confidence scoring, team intelligence, H2H analysis, projected goals, and AI-generated insider context.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-row">
      <div class="stat-block"><div class="stat-num">68%</div><div class="stat-unit">Accuracy</div><div class="stat-desc">Out-of-sample prediction accuracy on held-out test set</div></div>
      <div class="stat-block"><div class="stat-num">219</div><div class="stat-unit">Features</div><div class="stat-desc">Rolling averages, streaks, H2H ratios, goal patterns and more</div></div>
      <div class="stat-block"><div class="stat-num">10</div><div class="stat-unit">Seasons</div><div class="stat-desc">Trained on a decade of Premier League match data</div></div>
      <div class="stat-block"><div class="stat-num">60/40</div><div class="stat-unit">Ensemble</div><div class="stat-desc">XGBoost weighted at 60%, Random Forest at 40% for optimal blend</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="vs-section">
      <div class="sec-header" style="margin-top:7rem;">
        <div class="sec-label">THE DIFFERENCE</div><div class="sec-title">WHY KICKIQ HITS DIFFERENT</div>
      </div>
      <div class="compare-grid">
        <div class="compare-col ours">
          <div class="compare-col-label">⚡ KICKIQ</div>
          <div class="compare-row"><span class="c-check">✓</span>219 engineered features per match</div>
          <div class="compare-row"><span class="c-check">✓</span>XGBoost + Random Forest ensemble</div>
          <div class="compare-row"><span class="c-check">✓</span>10 years of training data</div>
          <div class="compare-row"><span class="c-check">✓</span>Home/Away split analysis</div>
          <div class="compare-row"><span class="c-check">✓</span>Head-to-head historical patterns</div>
          <div class="compare-row"><span class="c-check">✓</span>Momentum &amp; form streaks tracked</div>
          <div class="compare-row"><span class="c-check">✓</span>Clean sheet &amp; BTTS analysis</div>
          <div class="compare-row"><span class="c-check">✓</span>Confidence-scored signals</div>
          <div class="compare-row"><span class="c-check">✓</span>AI insider context per match</div>
        </div>
        <div class="compare-divider"><div class="vs-pill">VS</div></div>
        <div class="compare-col">
          <div class="compare-col-label" style="color:rgba(244,244,245,0.28);">Other Predictors</div>
          <div class="compare-row"><span class="c-cross">✗</span>Single model only</div>
          <div class="compare-row"><span class="c-cross">✗</span>Only recent form</div>
          <div class="compare-row"><span class="c-cross">✗</span>No ensemble stacking</div>
          <div class="compare-row"><span class="c-cross">✗</span>Basic win/loss stats</div>
          <div class="compare-row"><span class="c-cross">✗</span>No H2H weighting</div>
          <div class="compare-row"><span class="c-cross">✗</span>Static output only</div>
          <div class="compare-row"><span class="c-cross">✗</span>No defensive metrics</div>
          <div class="compare-row"><span class="c-cross">✗</span>No confidence scoring</div>
          <div class="compare-row"><span class="c-cross">✗</span>No contextual insights</div>
        </div>
      </div>
    </div>

    <!-- FEATURE BREAKDOWN -->
    <div class="sec-header">
      <div class="sec-label">UNDER THE HOOD</div>
      <div class="sec-title">KEY FEATURE CATEGORIES</div>
      <div class="sec-sub">What the models actually see</div>
    </div>
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon">📈</div>
        <div class="feat-title">FORM &amp; MOMENTUM</div>
        <div class="feat-desc">Rolling win/loss/draw rates, goal-scoring streaks, consecutive clean sheets, points-per-game over 5, 10, and 20-match windows.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">⚔️</div>
        <div class="feat-title">HEAD TO HEAD</div>
        <div class="feat-desc">Historical H2H win ratios, average goals in H2H meetings, home vs away performance in direct encounters, recency-weighted.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🏠</div>
        <div class="feat-title">VENUE SPLITS</div>
        <div class="feat-desc">Separate home and away form, goals scored and conceded at each venue, clean sheet rates home vs away.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">⚽</div>
        <div class="feat-title">GOAL PATTERNS</div>
        <div class="feat-desc">Average goals scored and conceded, BTTS rates, over/under 2.5 tendencies, first/second half goal split percentages.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🎯</div>
        <div class="feat-title">ATTACK VS DEFENCE</div>
        <div class="feat-desc">Offensive rating vs opposition defensive rating matchup, goal difference per window, xG proxies from historical shot data.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">📅</div>
        <div class="feat-title">TEMPORAL FEATURES</div>
        <div class="feat-desc">Seasonal position, fatigue proxies from fixture congestion, early vs late season trends, promotion/relegation pressure indicators.</div>
      </div>
    </div>

    <!-- FAQ -->
    <div class="sec-header" style="margin-top:6rem;">
      <div class="sec-label">COMMON QUESTIONS</div>
      <div class="sec-title">FAQ</div>
    </div>
    <div class="faq-list">
      <div class="faq-item">
        <div class="faq-q">How accurate is KICKIQ?</div>
        <div class="faq-a">The ensemble model achieves ~68% accuracy on out-of-sample test data — significantly above the ~46% baseline of always predicting the majority class (home win). Football is inherently unpredictable; this tool provides probabilistic guidance, not certainty.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">What data does it use?</div>
        <div class="faq-a">10 seasons of English Premier League match data including results, goals scored/conceded, and engineered form features. Data is from publicly available football statistics sources.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">Does it factor in injuries or suspensions?</div>
        <div class="faq-a">No. The model uses purely historical statistical patterns. Injuries, suspensions, weather conditions, referee tendencies, and team news are NOT modelled. Always apply contextual knowledge before acting on predictions.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">Can I use this for betting?</div>
        <div class="faq-a">KICKIQ is for entertainment purposes only. It does not constitute financial or betting advice. Gambling carries significant financial risk. Please gamble responsibly and within your means.</div>
      </div>
    </div>

    <div class="footer-line"></div>
    <div class="landing-footer">
      Built with XGBoost · scikit-learn · Streamlit · 2024/25 EPL Data<br>
      Model accuracy based on out-of-sample test set evaluation<br>
      ⚠ For entertainment purposes only · Not financial advice · Bet responsibly
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'predict':
    if not models_ok or not data_ok:
        st.error("Run `python src/train_models.py` first to generate models & features.")
        st.stop()

    st.markdown("""
    <div class="pred-topbar">
      <div><div class="pred-logo-text">KICK<span class="iq">IQ</span></div><div class="pred-subtitle"><span class="live-dot"></span>PREDICTION ENGINE · LIVE</div></div>
      <div class="topbar-tag">EPL 2024/25</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="back-wrap">', unsafe_allow_html=True)
    if st.button("← BACK", key="back_btn"):
        st.session_state.page='landing'; st.session_state.result=None; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="selector-zone">', unsafe_allow_html=True)
    st.markdown('<div style="max-width:960px;margin:auto;"><div class="section-kicker">MATCH SELECTION</div><div class="section-title" style="text-align:center;">WHO\'S PLAYING?</div></div>', unsafe_allow_html=True)
    _, center_area, _ = st.columns([1,7,1])
    with center_area:
        c1, c2, c3 = st.columns([5,1,5])
        with c1: home_team = st.selectbox("  HOME TEAM", teams, key="home_sel")
        with c2:
            st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;padding-top:6px;'><span style=\"font-family:'Anton';font-size:1.8rem;color:rgba(244,244,245,0.1);\">VS</span></div>", unsafe_allow_html=True)
        with c3:
            away_opts = [t for t in teams if t!=home_team]
            away_team = st.selectbox(" AWAY TEAM", away_opts, key="away_sel")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
    _, cta_m, _ = st.columns([2,4,2])
    with cta_m:
        clicked = st.button("⚡  ANALYSE THIS MATCH", key="pred_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        with st.spinner('Running ensemble analysis…'):
            st.session_state.result = run_prediction(home_team, away_team, xgb_model, rf_model, df, feature_cols)

    if st.session_state.result:
        res=st.session_state.result; hs=get_team_stats(df,home_team); as_=get_team_stats(df,away_team)
        hf=get_team_form(df,home_team,5); af=get_team_form(df,away_team,5); h2h=get_h2h(df,home_team,away_team,5)

        st.markdown('<div style="padding:0 4rem 6rem;position:relative;z-index:1;">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-hero">
          <div class="result-eyebrow">{home_team.upper()} vs {away_team.upper()} · {datetime.now().strftime('%d %B %Y').upper()}</div>
          <div class="result-outcome-text">{res['outcome'].upper()}</div>
          <div class="result-matchup-text">MODEL PREDICTION · ENSEMBLE</div>
          {signal_badge_html(res['confidence'])}
          <div class="disclaimer-txt">⚠ Entertainment only · Not financial advice · Bet at your own risk</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-tag" style="margin-top:2rem;">PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
        ha="active" if res['outcome']=='Home Win' else ""; da="active" if res['outcome']=='Draw' else ""; aa="active" if res['outcome']=='Away Win' else ""
        st.markdown(f"""
        <div class="prob-grid">
          <div class="prob-card {ha}"><div class="prob-pct">{res['proba'][0]*100:.0f}%</div><div class="prob-label">🏠 {home_team}</div></div>
          <div class="prob-card {da}"><div class="prob-pct">{res['proba'][1]*100:.0f}%</div><div class="prob-label">🤝 Draw</div></div>
          <div class="prob-card {aa}"><div class="prob-pct">{res['proba'][2]*100:.0f}%</div><div class="prob-label">✈️ {away_team}</div></div>
        </div>""", unsafe_allow_html=True)

        winner_idx=int(np.argmax(res['proba']))
        bar_colors=['rgba(200,255,0,0.85)' if i==winner_idx else 'rgba(255,255,255,0.06)' for i in range(3)]
        fig=go.Figure(go.Bar(x=[home_team,'Draw',away_team],y=[p*100 for p in res['proba']],marker=dict(color=bar_colors,line=dict(width=0),cornerradius=6),text=[f"{p*100:.1f}%" for p in res['proba']],textposition='outside',textfont=dict(family='DM Mono',size=11,color='rgba(244,244,245,0.4)')))
        fig.update_layout(height=200,margin=dict(l=0,r=0,t=18,b=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',yaxis=dict(showgrid=False,showticklabels=False,range=[0,115]),xaxis=dict(showgrid=False,tickfont=dict(family='DM Mono',size=10,color='rgba(244,244,245,0.28)')),bargap=0.38)
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

        st.markdown('<div class="line-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">TEAM INTELLIGENCE</div>', unsafe_allow_html=True)

        def gd_color(v): return "#C8FF00" if v>=0 else "#FF3A3A"
        def gd_str(v): return f"+{v}" if v>=0 else str(v)

        col1,col2=st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker"> {home_team.upper()} · RECENT FORM</div>
              <div class="intel-metrics">
                <div class="intel-metric"><div class="intel-metric-val">{hs['wins']}</div><div class="intel-metric-lbl">Wins</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{hs['draws']}</div><div class="intel-metric-lbl">Draws</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{hs['losses']}</div><div class="intel-metric-lbl">Losses</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{hs['win_rate']}%</div><div class="intel-metric-lbl">Win Rate</div></div>
              </div>
              <div class="intel-metrics">
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{hs['goals_for']}</div><div class="intel-metric-lbl">Goals For</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#FF3A3A;">{hs['goals_against']}</div><div class="intel-metric-lbl">Conceded</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:{gd_color(hs['gd'])};">{gd_str(hs['gd'])}</div><div class="intel-metric-lbl">Goal Diff</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{hs['clean_sheets']}</div><div class="intel-metric-lbl">Clean Sheets</div></div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.5rem;">LAST 5</div>
              {form_html(hf)}
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker"> {away_team.upper()} · RECENT FORM</div>
              <div class="intel-metrics">
                <div class="intel-metric"><div class="intel-metric-val">{as_['wins']}</div><div class="intel-metric-lbl">Wins</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{as_['draws']}</div><div class="intel-metric-lbl">Draws</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{as_['losses']}</div><div class="intel-metric-lbl">Losses</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{as_['win_rate']}%</div><div class="intel-metric-lbl">Win Rate</div></div>
              </div>
              <div class="intel-metrics">
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{as_['goals_for']}</div><div class="intel-metric-lbl">Goals For</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#FF3A3A;">{as_['goals_against']}</div><div class="intel-metric-lbl">Conceded</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:{gd_color(as_['gd'])};">{gd_str(as_['gd'])}</div><div class="intel-metric-lbl">Goal Diff</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{as_['clean_sheets']}</div><div class="intel-metric-lbl">Clean Sheets</div></div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.5rem;">LAST 5</div>
              {form_html(af)}
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-tag" style="margin-top:1.5rem;">MATCH INSIGHTS</div>', unsafe_allow_html=True)
        home_avg_scored=round(hs['goals_for']/max(hs['played'],1),2); away_avg_scored=round(as_['goals_for']/max(as_['played'],1),2)
        home_avg_conc=round(hs['goals_against']/max(hs['played'],1),2); away_avg_conc=round(as_['goals_against']/max(as_['played'],1),2)
        exp_home=round((home_avg_scored+away_avg_conc)/2,2); exp_away=round((away_avg_scored+home_avg_conc)/2,2)
        h_cls='good' if home_avg_scored>away_avg_scored else ''; a_cls='good' if away_avg_scored>home_avg_scored else ''
        h_exp_cls='good' if exp_home>exp_away else 'bad' if exp_home<exp_away else ''
        a_exp_cls='good' if exp_away>exp_home else 'bad' if exp_away<exp_home else ''
        st.markdown(f"""
        <div class="extra-insight">
          <div class="intel-card-kicker">🔭 PROJECTED SCORING BREAKDOWN</div>
          <div class="insight-row">
            <div class="insight-item"><div class="insight-item-label"> {home_team} Avg Scored</div><div class="insight-item-val {h_cls}">{home_avg_scored} per game</div></div>
            <div class="insight-item"><div class="insight-item-label"> {away_team} Avg Scored</div><div class="insight-item-val {a_cls}">{away_avg_scored} per game</div></div>
            <div class="insight-item"><div class="insight-item-label"> {home_team} Proj Goals</div><div class="insight-item-val {h_exp_cls}">{exp_home}</div></div>
            <div class="insight-item"><div class="insight-item-label"> {away_team} Proj Goals</div><div class="insight-item-val {a_exp_cls}">{exp_away}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        if h2h:
            st.markdown('<div class="section-tag" style="margin-top:0.5rem;">HEAD TO HEAD</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="h2h-card">
              <div class="intel-card-kicker"> LAST {len(h2h)} MEETINGS · {home_team.upper()} PERSPECTIVE</div>
              {form_html(h2h)}
              <div class="h2h-stat"><span></span><span>{home_team} won {h2h.count('W')}, drew {h2h.count('D')}, lost {h2h.count('L')} of last {len(h2h)} meetings against {away_team}.</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-tag" style="margin-top:0.5rem;">INSIDER INTEL</div>', unsafe_allow_html=True)
        notes=insider_notes(home_team,away_team,res,hs,as_,h2h)
        notes_html=''.join(f'<div class="notes-item"><span class="note-icon">{ic}</span><span>{tx}</span></div>' for ic,tx in notes)
        st.markdown(f"""
        <div class="notes-card">
          <div class="intel-card-kicker"> AI ANALYSIS · {home_team.upper()} vs {away_team.upper()}</div>
          {notes_html}
        </div>""", unsafe_allow_html=True)

        with st.expander("  MODEL INTERNALS — XGBoost vs Random Forest"):
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**XGBoost (60% weight)**")
                for label,p in zip([home_team,'Draw',away_team],res['xgb']): st.markdown(f"`{label}` → **{p*100:.1f}%**")
            with c2:
                st.markdown("**Random Forest (40% weight)**")
                for label,p in zip([home_team,'Draw',away_team],res['rf']): st.markdown(f"`{label}` → **{p*100:.1f}%**")

        st.markdown('</div>', unsafe_allow_html=True)
