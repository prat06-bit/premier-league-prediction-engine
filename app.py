import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from math import factorial, exp

st.set_page_config(
    page_title="KICKIQ · EPL Predictor",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── FONT LOADING ──────────────────────────────────────────────────────────────
# Must be a separate st.markdown call BEFORE the <style> block.
# On Streamlit Cloud, @import inside injected CSS is unreliable because the
# browser may parse the <style> before the @import URL finishes loading.
# <link rel="stylesheet"> loads fonts in parallel before any CSS is applied.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&family=DM+Mono:wght@300;400;500&family=Rajdhani:wght@300;400;500;600;700&family=Saira+Condensed:wght@200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fonts loaded via <link> tags above — @import removed for Cloud reliability */

*,*::before,*::after{box-sizing:border-box;}
:root{
  --acid:#C8FF00;--void:#050608;--surface:#0A0C0F;--surface2:#0F1115;
  --border:rgba(255,255,255,0.06);--text:#F4F4F5;
  --muted:rgba(244,244,245,0.4);--muted2:rgba(244,244,245,0.22);
  --red:#FF3A3A;--amber:#FFB800;--cyan:#00E5FF;
}
html,body,.stApp{background:var(--void) !important;color:var(--text);font-family:'Rajdhani',sans-serif;overflow-x:hidden;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:var(--void);}::-webkit-scrollbar-thumb{background:var(--acid);border-radius:2px;}

body::after{content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.028'/%3E%3C/svg%3E");
  pointer-events:none;z-index:9999;mix-blend-mode:overlay;}

/* Streamlit chrome removal */
header[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stToolbar"],
#MainMenu,footer{display:none !important;height:0 !important;}
.block-container,.stMainBlockContainer,[data-testid="stAppViewBlockContainer"],
section.main > div,div.appview-container section.main > div:first-child{
  padding:0 !important;margin:0 !important;gap:0 !important;
  max-width:100% !important;width:100% !important;}
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"]{gap:0 !important;padding:0 !important;margin:0 !important;}

/* ── GLOBAL BUTTON BASE ── */
div[data-testid="stButton"]{display:flex !important;justify-content:center !important;width:100% !important;}
div[data-testid="stButton"] > button{
  background:var(--acid) !important;color:var(--void) !important;border:none !important;
  border-radius:12px !important;font-family:'Anton',sans-serif !important;
  font-size:1.1rem !important;letter-spacing:0.1em !important;
  padding:0.95rem 2.5rem !important;min-width:220px !important;width:auto !important;
  /* margin:0 auto ensures self-centering even when flex parent misbehaves on Cloud */
  margin:0 auto !important;
  cursor:pointer !important;transition:all 0.22s cubic-bezier(0.34,1.56,0.64,1) !important;
  animation:ctaPulse 4s ease-in-out infinite !important;}
div[data-testid="stButton"] > button:hover{
  transform:translateY(-3px) scale(1.02) !important;
  box-shadow:0 0 50px rgba(200,255,0,0.6),0 8px 30px rgba(200,255,0,0.25) !important;
  animation:none !important;}
div[data-testid="stButton"] > button:active{transform:scale(0.98) !important;}

/* ── BACK BUTTON ─────────────────────────────────────────────────────────────
   ROOT CAUSE FIX: st.markdown('<div class="kiq-nav-back">') renders as an empty
   sibling element in the DOM — NOT as a parent of st.button(). So the old
   ".kiq-nav-back div[stButton]" selector never matched.
   FIX: :has() looks DOWN into descendants of the column's stVerticalBlock,
   finding the .kiq-nav-back marker and scoping all overrides to that block. ── */

/* Shrink the column stVerticalBlock that hosts the back button */
[data-testid="stVerticalBlock"]:has(.kiq-nav-back),
[data-testid="stVerticalBlock"]:has(.kiq-nav-back) [data-testid="stVerticalBlockBorderWrapper"] {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}
/* Override the global button container inside the back-button column */
[data-testid="stVerticalBlock"]:has(.kiq-nav-back) [data-testid="stButton"] {
  justify-content: flex-start !important;
  width: auto !important;
  min-width: 0 !important;
}
/* Override the actual button element — kill global min-width and padding */
[data-testid="stVerticalBlock"]:has(.kiq-nav-back) [data-testid="stButton"] > button {
  background: var(--acid) !important;
  color: var(--void) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  padding: 0.3rem 0.8rem !important;
  border-radius: 6px !important;
  min-width: 0 !important;
  width: auto !important;
  max-width: 110px !important;
  margin: 0 !important;           /* cancel global margin:0 auto for this button */
  animation: none !important;
  white-space: nowrap !important;
  line-height: 1.5 !important;
}
[data-testid="stVerticalBlock"]:has(.kiq-nav-back) [data-testid="stButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 0 16px rgba(200,255,0,0.4) !important;
}

/* ── NAVBAR ──────────────────────────────────────────────────────────────────
   ROOT CAUSE FIX: .kiq-nav-row is an empty sibling div, NOT a wrapper of the
   stHorizontalBlock. The old ".kiq-nav-row [data-testid='stHorizontalBlock']"
   rule had ZERO matching elements in the live DOM.
   
   FIX 1 (header cropping): Hide .kiq-nav-row so the empty auto-closed div
   doesn't consume layout space and crop the nav bar.
   FIX 2 (sticky nav): Use adjacent-sibling combinator (+) to target the
   stHorizontalBlock that immediately follows the element-container holding
   the .kiq-nav-row marker. This is a direct sibling relationship in the
   Streamlit stVerticalBlock, so it always matches correctly. ── */

/* Hide the empty marker — removes phantom height causing header cropping */
.kiq-nav-row { display: none !important; }

/* Style the actual nav stHorizontalBlock via adjacent-sibling after marker */
[data-testid="element-container"]:has(.kiq-nav-row) + [data-testid="stHorizontalBlock"] {
  position: sticky !important;
  top: 0 !important;
  z-index: 300 !important;
  background: rgba(5,6,8,0.97) !important;
  border-bottom: 1px solid rgba(200,255,0,0.08) !important;
  min-height: 62px !important;
  padding: 0 1.5rem !important;
  box-sizing: border-box !important;
  align-items: center !important;
  width: 100% !important;
  gap: 0 !important;
  overflow: visible !important;
}

/* ── ORBS & GRID ── */
.orb-container{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(90px);animation:orbFloat var(--dur,22s) ease-in-out infinite;}
.orb-1{width:600px;height:600px;top:-15%;left:-10%;background:radial-gradient(circle,rgba(200,255,0,0.1) 0%,transparent 70%);--dur:20s;}
.orb-2{width:450px;height:450px;bottom:-10%;right:-8%;background:radial-gradient(circle,rgba(0,229,255,0.07) 0%,transparent 70%);--dur:25s;animation-delay:-8s;animation-direction:reverse;}
.orb-3{width:350px;height:350px;top:50%;left:55%;background:radial-gradient(circle,rgba(200,255,0,0.04) 0%,transparent 70%);--dur:32s;animation-delay:-14s;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}33%{transform:translate(25px,-35px) scale(1.04);}66%{transform:translate(-18px,28px) scale(0.96);}}
.bg-grid{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(200,255,0,0.022) 1px,transparent 1px),linear-gradient(90deg,rgba(200,255,0,0.022) 1px,transparent 1px);
  background-size:80px 80px;
  mask-image:radial-gradient(ellipse 75% 75% at 50% 50%,black 25%,transparent 100%);}

/* ── LANDING ── */
.landing-wrap{width:100%;display:flex;flex-direction:column;align-items:center;padding:5vh 1rem 0;position:relative;z-index:1;}
.eyebrow{display:inline-flex;align-items:center;gap:10px;background:linear-gradient(135deg,rgba(200,255,0,0.08),rgba(200,255,0,0.02));border:1px solid rgba(200,255,0,0.2);border-radius:100px;padding:6px 18px 6px 10px;font-family:'DM Mono',monospace;font-size:0.66rem;letter-spacing:0.18em;color:var(--acid);text-transform:uppercase;margin-bottom:2rem;animation:fadeDown 0.6s cubic-bezier(0.16,1,0.3,1) 0.05s both;}
.eyebrow-pulse{width:7px;height:7px;background:var(--acid);border-radius:50%;animation:pls 1.8s ease-in-out infinite;box-shadow:0 0 8px var(--acid);}
@keyframes pls{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.3;transform:scale(0.65);}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-12px);}to{opacity:1;transform:translateY(0);}}
.hero-wordmark{font-family:'Anton',sans-serif;font-size:clamp(5.5rem,16vw,13rem);line-height:0.85;letter-spacing:-0.02em;text-align:center;position:relative;z-index:1;animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.1s both;}
.wm-kick{color:var(--text);}.wm-iq{color:var(--acid);text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);animation:acidFlicker 6s ease-in-out infinite 2.5s;}
@keyframes heroUp{from{opacity:0;transform:translateY(28px) skewY(1.5deg);}to{opacity:1;transform:translateY(0) skewY(0);}}
@keyframes acidFlicker{0%,88%,100%{text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);}90%{text-shadow:0 0 6px rgba(200,255,0,0.2);opacity:0.85;}92%{text-shadow:0 0 60px rgba(200,255,0,0.5),0 0 120px rgba(200,255,0,0.2);}95%{text-shadow:0 0 6px rgba(200,255,0,0.2);opacity:0.78;}}
.hero-tagline{font-family:'Saira Condensed',sans-serif;font-weight:300;font-size:clamp(0.9rem,2vw,1.1rem);letter-spacing:0.42em;text-transform:uppercase;color:var(--muted);text-align:center;margin:1.4rem 0 0;animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.22s both;}
.metrics-strip{display:flex;gap:0;align-items:stretch;margin:3.5rem auto 0;width:100%;max-width:700px;border:1px solid var(--border);border-radius:16px;overflow:hidden;background:var(--surface);animation:heroUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.38s both;}
.metric-block{flex:1;padding:1.8rem 2rem;position:relative;border-right:1px solid var(--border);text-align:center;transition:background 0.3s;}
.metric-block:last-child{border-right:none;}.metric-block:hover{background:rgba(200,255,0,0.03);}
.metric-block::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--acid),transparent);transform:scaleX(0);transition:transform 0.35s;}
.metric-block:hover::after{transform:scaleX(1);}
.metric-val{font-family:'Anton',sans-serif;font-size:2.8rem;line-height:1;color:var(--acid);}
.metric-lbl{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin-top:5px;}
.sec-header{text-align:center;margin:7rem 0 3.5rem;position:relative;z-index:1;}
.sec-label{font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.32em;color:var(--acid);text-transform:uppercase;margin-bottom:0.8rem;}
.sec-title{font-family:'Anton',sans-serif;font-size:clamp(2.5rem,5vw,4rem);letter-spacing:0.01em;color:var(--text);line-height:1;}
.sec-sub{font-family:'Saira Condensed',sans-serif;font-weight:300;font-size:1rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase;margin-top:0.8rem;}
.hiw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;width:100%;max-width:900px;margin:0 auto;position:relative;z-index:1;}
.hiw-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:2rem 1.8rem;position:relative;overflow:hidden;transition:transform 0.3s,border-color 0.3s,box-shadow 0.3s;opacity:0;animation:revealUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards;}
.hiw-card:nth-child(1){animation-delay:0.1s;}.hiw-card:nth-child(2){animation-delay:0.22s;}.hiw-card:nth-child(3){animation-delay:0.34s;}
.hiw-card:hover{transform:translateY(-5px);border-color:rgba(200,255,0,0.2);box-shadow:0 12px 50px rgba(0,0,0,0.5),0 0 25px rgba(200,255,0,0.06);}
.hiw-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(200,255,0,0.3),transparent);opacity:0;transition:opacity 0.3s;}
.hiw-card:hover::before{opacity:1;}
.hiw-num{font-family:'Anton',sans-serif;font-size:4rem;line-height:1;color:rgba(200,255,0,0.07);margin-bottom:1rem;}
.hiw-title{font-family:'Anton',sans-serif;font-size:1.3rem;letter-spacing:0.04em;color:var(--text);margin-bottom:0.6rem;}
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
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;width:100%;max-width:900px;margin:0 auto;position:relative;z-index:1;}
.feat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.6rem 1.5rem;transition:border-color 0.3s,transform 0.3s;animation:revealUp 0.5s cubic-bezier(0.16,1,0.3,1) both;}
.feat-card:hover{border-color:rgba(200,255,0,0.18);transform:translateY(-3px);}
.feat-title{font-family:'Anton',sans-serif;font-size:1rem;letter-spacing:0.05em;color:var(--text);margin-bottom:0.5rem;}
.feat-desc{font-family:'Rajdhani',sans-serif;font-size:0.88rem;font-weight:400;color:var(--muted);line-height:1.55;}
.faq-list{width:100%;max-width:760px;margin:0 auto;position:relative;z-index:1;}
.faq-item{border-bottom:1px solid rgba(255,255,255,0.05);padding:1.6rem 0;animation:revealUp 0.5s cubic-bezier(0.16,1,0.3,1) both;}
.faq-item:last-child{border-bottom:none;}
.faq-q{font-family:'Anton',sans-serif;font-size:1.1rem;letter-spacing:0.03em;color:var(--text);margin-bottom:0.6rem;display:flex;align-items:center;gap:10px;}
.faq-q::before{content:'';width:4px;height:1.1rem;background:var(--acid);border-radius:2px;flex-shrink:0;}
.faq-a{font-family:'Rajdhani',sans-serif;font-size:0.95rem;font-weight:400;color:var(--muted);line-height:1.65;padding-left:14px;}

/* ── PREDICT PAGE ── */
.kiq-hero{width:100%;text-align:center;padding:2.5rem 1rem 1.2rem;position:relative;z-index:1;}
.match-kicker{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.32em;color:rgba(200,255,0,0.6);text-transform:uppercase;margin-bottom:0.6rem;display:flex;align-items:center;justify-content:center;gap:12px;}
.match-kicker::before,.match-kicker::after{content:'';width:28px;height:1px;background:rgba(200,255,0,0.28);}
.match-title{font-family:'Anton',sans-serif;font-size:clamp(3rem,8vw,6.5rem);letter-spacing:-0.01em;color:#F4F4F5;line-height:0.88;text-align:center;margin-bottom:0.35rem;}
.match-sub{font-family:'DM Mono',monospace;font-size:0.48rem;letter-spacing:0.2em;color:rgba(244,244,245,0.16);text-transform:uppercase;text-align:center;margin-bottom:1.4rem;}

/* ── PICKER CARD via :has() — targets the real Streamlit stHorizontalBlock
   that contains selectboxes, no HTML wrapper div needed ── */
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) {
  background:linear-gradient(145deg,rgba(200,255,0,0.04) 0%,rgba(8,10,13,0.98) 100%) !important;
  border:1px solid rgba(200,255,0,0.18) !important;
  border-radius:18px !important;
  padding:1.8rem 2rem 1.5rem !important;
  box-shadow:0 0 50px rgba(200,255,0,0.04),0 20px 60px rgba(0,0,0,0.5) !important;
  position:relative !important;
  overflow:visible !important;
  width:100% !important;
  max-width:860px !important;
  margin:0 auto 1.4rem !important;
  box-sizing:border-box !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"])::before{
  content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(200,255,0,0.35),transparent);}
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) .stSelectbox label{
  font-family:'DM Mono',monospace !important;font-size:0.52rem !important;
  letter-spacing:0.22em !important;color:rgba(244,244,245,0.32) !important;
  text-transform:uppercase !important;margin-bottom:0.4rem !important;display:block !important;}
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) .stSelectbox > div > div{
  background:rgba(4,5,7,0.85) !important;border:1px solid rgba(200,255,0,0.16) !important;
  border-radius:10px !important;
  font-size:1.1rem !important;padding:0.7rem 1rem !important;
  transition:border-color 0.2s !important;box-shadow:none !important;}
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) .stSelectbox > div > div:hover{
  border-color:rgba(200,255,0,0.45) !important;}

/* ── SELECTBOX TEXT — belt-and-suspenders fallback ───────────────────────────
   PRIMARY FIX: .streamlit/config.toml sets textColor=#F4F4F5 which BaseWeb
   reads at render time — this makes every widget text correct automatically.
   The rules below are fallbacks in case config.toml is not picked up yet. ── */

/* Value container — the div that holds the selected text */
.stSelectbox [data-baseweb="select"] [data-baseweb="value"],
.stSelectbox [data-baseweb="select"] [data-baseweb="value"] div,
.stSelectbox [data-baseweb="select"] [data-baseweb="value"] span {
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  opacity: 1 !important;
}
/* Placeholder text */
.stSelectbox [data-baseweb="select"] [data-baseweb="placeholder"] {
  color: var(--muted) !important;
  -webkit-text-fill-color: var(--muted) !important;
}
/* Keep the dropdown arrow acid-colored */
.stSelectbox [data-baseweb="select"] svg {
  fill: rgba(200,255,0,0.5) !important;
  color: rgba(200,255,0,0.5) !important;
}
/* Dim placeholder text — keep -webkit-text-fill-color consistent */
.stSelectbox [data-baseweb="select"] [data-baseweb="placeholder"],
.stSelectbox [data-baseweb="select"] input::placeholder {
  color: rgba(244,244,245,0.25) !important;
  -webkit-text-fill-color: rgba(244,244,245,0.25) !important;
}
/* Dropdown list items */
ul[data-baseweb="menu"] li,
ul[data-baseweb="menu"] li * {
  color: var(--text) !important;
  background: #0A0C0F !important;
  font-family: 'Rajdhani', sans-serif !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li:hover * {
  background: rgba(200,255,0,0.08) !important;
  color: var(--acid) !important;
}

.vs-badge{font-family:'Anton',sans-serif;font-size:0.95rem;color:rgba(244,244,245,0.12);letter-spacing:0.1em;text-align:center;padding-top:1.9rem;}
.pred-meta{font-family:'DM Mono',monospace;font-size:0.44rem;letter-spacing:0.15em;color:rgba(244,244,245,0.12);text-transform:uppercase;text-align:center;margin-top:0.6rem;}
.pred-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(200,255,0,0.2),transparent);margin:2.2rem 0;}

/* ── RESULTS ── */
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
.prob-card:hover{transform:translateY(-5px) scale(1.02);border-color:rgba(200,255,0,0.2);}
.prob-card.active{background:linear-gradient(135deg,rgba(200,255,0,0.1),rgba(200,255,0,0.04));border-color:rgba(200,255,0,0.38);box-shadow:0 0 35px rgba(200,255,0,0.1);}
.prob-card.active::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--acid),transparent);}
.prob-pct{font-family:'Anton',sans-serif;font-size:3rem;line-height:1;color:var(--text);transition:color 0.3s;}
.prob-card.active .prob-pct{color:var(--acid);}
.prob-label{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin-top:6px;}
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
.notes-item:last-child{border-bottom:none;}
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

/* Scoreline table */
.scoreline-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:0.5rem;margin-top:1rem;}
.scoreline-cell{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.7rem 0.5rem;text-align:center;}
.scoreline-cell.top{border-color:rgba(200,255,0,0.3);background:rgba(200,255,0,0.07);}
.scoreline-val{font-family:'Anton',sans-serif;font-size:1.1rem;color:var(--text);}
.scoreline-cell.top .scoreline-val{color:var(--acid);}
.scoreline-pct{font-family:'DM Mono',monospace;font-size:0.48rem;letter-spacing:0.1em;color:var(--muted);text-transform:uppercase;margin-top:3px;}

/* Venue split bars */
.venue-bar-wrap{margin-top:0.8rem;}
.venue-bar-label{font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:0.12em;color:var(--muted);text-transform:uppercase;margin-bottom:5px;display:flex;justify-content:space-between;}
.venue-bar-track{height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;}
.venue-bar-fill{height:100%;border-radius:3px;transition:width 1.2s cubic-bezier(0.16,1,0.3,1);}

/* Streak badge */
.streak-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:6px;font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;font-weight:500;margin-top:0.6rem;}
.streak-W{background:rgba(200,255,0,0.12);border:1px solid rgba(200,255,0,0.25);color:var(--acid);}
.streak-D{background:rgba(255,184,0,0.12);border:1px solid rgba(255,184,0,0.22);color:var(--amber);}
.streak-L{background:rgba(255,58,58,0.1);border:1px solid rgba(255,58,58,0.18);color:var(--red);}

/* Market odds-style bar */
.odds-row{display:flex;height:48px;border-radius:10px;overflow:hidden;margin:1rem 0;border:1px solid rgba(255,255,255,0.04);}
.odds-seg{display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.08em;font-weight:500;transition:flex 1s cubic-bezier(0.16,1,0.3,1);}
.odds-seg.home{background:rgba(200,255,0,0.18);color:var(--acid);}
.odds-seg.draw{background:rgba(255,184,0,0.12);color:var(--amber);}
.odds-seg.away{background:rgba(0,229,255,0.12);color:var(--cyan);}

.stSpinner>div{border-top-color:var(--acid) !important;}
.streamlit-expanderHeader{font-family:'DM Mono',monospace !important;font-size:0.66rem !important;letter-spacing:0.15em !important;color:var(--muted) !important;background:var(--surface) !important;border:1px solid var(--border) !important;border-radius:10px !important;}
.streamlit-expanderContent{background:var(--surface) !important;border:1px solid var(--border) !important;border-top:none !important;}

/* Button using use_container_width=True — fills column without clipping */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button {
  display: block !important;
  text-align: center !important;
}
/* Ensure button columns are never overflow:hidden */
[data-testid="column"] {
  overflow: visible !important;
}

@keyframes ctaPulse{0%,100%{box-shadow:0 0 0 rgba(200,255,0,0);}50%{box-shadow:0 0 38px rgba(200,255,0,0.38);}}

@media(max-width:768px){
  .hero-wordmark{font-size:4.5rem;}
  .metrics-strip{flex-direction:column;}
  .hiw-grid,.feat-grid{grid-template-columns:1fr;}
  .stats-row{grid-template-columns:1fr 1fr;}
  .compare-grid{grid-template-columns:1fr;}
  .compare-divider{display:none;}
  .prob-grid{grid-template-columns:1fr 1fr;}
  .prob-grid .prob-card:last-child{grid-column:1/-1;}
  .insight-row{grid-template-columns:1fr;}
  .scoreline-grid{grid-template-columns:repeat(3,1fr);}
}
</style>
""", unsafe_allow_html=True)


# ── DATA & MODEL LOADING ──────────────────────────────────────────────────────
import warnings
import sklearn
import xgboost

def _get_env_versions() -> dict:
    """Return a dict of the currently installed library versions."""
    return {
        "sklearn":  sklearn.__version__,
        "xgboost":  xgboost.__version__,
        "python":   f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    }

def _parse_version(v: str) -> tuple:
    """Convert '1.6.1' → (1, 6, 1) for numeric comparison."""
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except Exception:
        return (0, 0, 0)

@st.cache_resource
def load_models():
    """
    Production-safe model loader.

    Strategy
    --------
    * Suppress the noisy-but-harmless InconsistentVersionWarning that sklearn
      emits when the pkl was created with a *newer* minor version than the
      installed one (e.g. trained on 1.8.0, running on 1.6.1).
    * Log the version delta so it is always visible in Streamlit Cloud logs.
    * Hard-fail only on a MAJOR version mismatch (1.x vs 2.x) — that is the
      only scenario where the internal object format genuinely breaks.
    * For XGBoost, warn but continue: XGBoost's own booster handles cross-
      version pkl gracefully unless the major version changes.
    """
    env = _get_env_versions()

    try:
        # ── Suppress minor-version warnings so they don't appear in the UI ──
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,          # XGBoost serialization warning
                message=".*older version.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=sklearn.exceptions.InconsistentVersionWarning,
            )

            xgb_model = joblib.load('models/tuned/xgboost_tuned.pkl')
            rf_model  = joblib.load('models/tuned/random_forest_tuned.pkl')
            fc        = joblib.load('models/tuned/feature_columns.pkl')

        # ── Version delta check: sklearn ──────────────────────────────────
        # sklearn embeds __getstate__ metadata on every estimator.
        # A safe way to read the train-time version without re-serialising:
        train_sklearn = None
        try:
            # RandomForest is a sklearn object — check its embedded version tag
            train_sklearn = getattr(rf_model, "_sklearn_version", None)
        except Exception:
            pass

        if train_sklearn:
            env_major  = _parse_version(env["sklearn"])[0]
            train_major = _parse_version(train_sklearn)[0]
            # Log version info to the Cloud console (visible in logs, not in UI)
            print(
                f"[KickIQ] sklearn  — trained:{train_sklearn}  "
                f"running:{env['sklearn']}  "
                f"{'⚠ MINOR DELTA' if train_sklearn != env['sklearn'] else '✓ MATCH'}"
            )
            if train_major != env_major:
                st.error(
                    f"⛔ sklearn MAJOR version mismatch: models trained on "
                    f"v{train_sklearn}, running v{env['sklearn']}. "
                    f"Re-train models with the current version or pin "
                    f"`scikit-learn=={env['sklearn']}` in requirements.txt."
                )
                return None, None, None, False
        else:
            print(
                f"[KickIQ] sklearn  — train-time version unknown  "
                f"running:{env['sklearn']}"
            )

        # ── Version delta check: xgboost ──────────────────────────────────
        print(
            f"[KickIQ] xgboost  — running:{env['xgboost']} "
            f"(train-time version not embedded in pkl)"
        )

        # ── Quick smoke-test: can the models actually predict? ─────────────
        try:
            n_features = len(fc)
            dummy = pd.DataFrame(
                [[0] * n_features], columns=fc
            )
            xgb_model.predict_proba(dummy)
            rf_model.predict_proba(dummy)
        except Exception as smoke_err:
            st.error(
                f"⛔ Model smoke-test failed: {smoke_err}. "
                f"The models are likely incompatible with the current library "
                f"versions. Re-train and re-save the models."
            )
            return None, None, None, False

        print(f"[KickIQ] Models loaded and smoke-tested ✓  env={env}")
        return xgb_model, rf_model, fc, True

    except FileNotFoundError as e:
        st.error(
            f"Model file not found: {e}. "
            f"Run `python src/train_models.py` to generate model files, "
            f"then commit them to the repo."
        )
        return None, None, None, False
    except Exception as e:
        st.error(f"Unexpected model load error: {e}")
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


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_team_form(df, team, n=5):
    home = df[df['HomeTeam']==team][['Date','FTR']].copy()
    home['result'] = home['FTR'].map({'H':'W','D':'D','A':'L'})
    away = df[df['AwayTeam']==team][['Date','FTR']].copy()
    away['result'] = away['FTR'].map({'H':'L','D':'D','A':'W'})
    combined = pd.concat([home[['Date','result']], away[['Date','result']]])
    return combined.sort_values('Date', ascending=False).head(n)['result'].tolist()

def get_streak(df, team):
    form = get_team_form(df, team, 20)
    if not form: return ('N', 0)
    cur = form[0]; cnt = 0
    for r in form:
        if r == cur: cnt += 1
        else: break
    return (cur, cnt)

def get_team_stats(df, team):
    home = df[df['HomeTeam']==team].copy()
    away = df[df['AwayTeam']==team].copy()
    all_m = pd.concat([
        home[['Date','FTHG','FTAG','FTR']].assign(venue='home'),
        away[['Date','FTHG','FTAG','FTR']].assign(venue='away')
    ]).sort_values('Date', ascending=False).head(38)
    if len(all_m)==0:
        return {'played':0,'wins':0,'draws':0,'losses':0,'goals_for':0,'goals_against':0,
                'gd':0,'win_rate':0,'clean_sheets':0,'btts':0,'ppg':0,
                'home_wins':0,'home_played':0,'away_wins':0,'away_played':0,
                'home_gf':0,'home_ga':0,'away_gf':0,'away_ga':0}
    wins   = (((all_m['venue']=='home')&(all_m['FTR']=='H'))|((all_m['venue']=='away')&(all_m['FTR']=='A'))).sum()
    draws  = (all_m['FTR']=='D').sum()
    losses = len(all_m)-wins-draws
    gf = int(np.where(all_m['venue']=='home', all_m['FTHG'], all_m['FTAG']).sum())
    ga = int(np.where(all_m['venue']=='home', all_m['FTAG'], all_m['FTHG']).sum())
    clean_sheets = int((np.where(all_m['venue']=='home', all_m['FTAG'], all_m['FTHG'])==0).sum())
    btts = int(((all_m['FTHG']>0)&(all_m['FTAG']>0)).sum())
    pts  = int(wins*3 + draws)
    ppg  = round(pts/len(all_m),2)
    hm = all_m[all_m['venue']=='home']
    aw = all_m[all_m['venue']=='away']
    hw = int((hm['FTR']=='H').sum())
    aaw= int((aw['FTR']=='A').sum())
    hgf= int(hm['FTHG'].sum()); hga= int(hm['FTAG'].sum())
    agf= int(aw['FTAG'].sum()); aga= int(aw['FTHG'].sum())
    return {'played':len(all_m),'wins':int(wins),'draws':int(draws),'losses':int(losses),
            'goals_for':gf,'goals_against':ga,'gd':gf-ga,'win_rate':round(wins/len(all_m)*100),
            'clean_sheets':clean_sheets,'btts':btts,'ppg':ppg,
            'home_wins':hw,'home_played':len(hm),'away_wins':aaw,'away_played':len(aw),
            'home_gf':hgf,'home_ga':hga,'away_gf':agf,'away_ga':aga}

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

def poisson_prob(lam, k):
    return (exp(-lam) * lam**k) / factorial(k)

def get_score_probs(exp_h, exp_a, max_g=5):
    """Return matrix of scoreline probabilities and aggregated stats."""
    matrix = {}
    for i in range(max_g+1):
        for j in range(max_g+1):
            matrix[(i,j)] = poisson_prob(exp_h, i) * poisson_prob(exp_a, j)
    total = sum(matrix.values())
    # normalise to visible range only
    matrix = {k: v/total for k, v in matrix.items()}

    sorted_scores = sorted(matrix.items(), key=lambda x: -x[1])[:5]

    btts = sum(v for (i,j),v in matrix.items() if i>0 and j>0)
    over25 = sum(v for (i,j),v in matrix.items() if i+j>2)
    over15 = sum(v for (i,j),v in matrix.items() if i+j>1)
    under25 = 1 - over25
    return {
        'top_scores': sorted_scores,
        'btts': round(btts*100),
        'over25': round(over25*100),
        'over15': round(over15*100),
        'under25': round(under25*100),
    }

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
    xp = xgb.predict_proba(feats)[0]
    rp = rf.predict_proba(feats)[0]
    ens = 0.6*xp + 0.4*rp
    return {'proba':ens,'outcome':['Home Win','Draw','Away Win'][np.argmax(ens)],
            'confidence':ens.max(),'xgb':xp,'rf':rp}

def form_html(form):
    return '<div class="form-row">'+''.join(
        f'<span class="fp fp-{r}">{r}</span>' for r in form)+'</div>'

def signal_badge_html(conf):
    if conf>=0.65: return '<div class="signal-badge s-strong">STRONG SIGNAL</div>'
    elif conf>=0.55: return '<div class="signal-badge s-mod">MODERATE SIGNAL</div>'
    elif conf>=0.45: return '<div class="signal-badge s-weak">WEAK SIGNAL</div>'
    else: return '<div class="signal-badge s-none">NO CLEAR EDGE</div>'

def insider_notes(home, away, res, hs, as_, h2h):
    notes = []
    idx = np.argmax(res['proba'])
    notes.append(f"Ensemble: XGBoost ({res['xgb'][idx]*100:.0f}%) + Random Forest ({res['rf'][idx]*100:.0f}%) weighted 60/40.")
    if res['confidence']>0.60:
        notes.append(f"High conviction — {res['outcome']} is the dominant call across both models.")
    else:
        notes.append("Probabilities are spread — contested fixture, upsets are very possible.")
    gd_diff = hs['gd']-as_['gd']
    if abs(gd_diff)>8:
        better = home if gd_diff>0 else away
        notes.append(f"{better} hold significantly stronger goal difference — superior attacking and defensive quality.")
    if hs['win_rate']>60:
        notes.append(f"{home} winning {hs['win_rate']}% of recent matches — elite form over the sample window.")
    if as_['win_rate']>60:
        notes.append(f"{away} carry {as_['win_rate']}% win rate — dangerous regardless of venue.")
    if hs['ppg']>=2.2:
        notes.append(f"{home} averaging {hs['ppg']} points per game — championship-level consistency.")
    if as_['ppg']>=2.2:
        notes.append(f"{away} averaging {as_['ppg']} points per game — top-tier efficiency on the road.")
    if hs['clean_sheets']>=5:
        notes.append(f"{home} kept {hs['clean_sheets']} clean sheets recently — rock-solid defensive unit.")
    if as_['clean_sheets']>=5:
        notes.append(f"{away} boast {as_['clean_sheets']} clean sheets — expect a tight game.")
    if h2h:
        hw=h2h.count('W'); hd=h2h.count('D'); hl=h2h.count('L')
        verdict = 'History favours the home side.' if hw>hl else 'Away side historically dominant.' if hl>hw else 'Historically level — anyone can win.'
        notes.append(f"H2H record ({home}): {hw}W / {hd}D / {hl}L from last {len(h2h)} meetings. {verdict}")
    if res['proba'][1]>0.30:
        notes.append(f"Draw probability at {res['proba'][1]*100:.0f}% — cagey, tactical fixture expected.")
    if hs['btts']>=6 and as_['btts']>=6:
        notes.append("Both teams to score is very likely — both sides regularly feature in high-scoring games.")
    notes.append("Injuries, suspensions, weather and referee tendencies are NOT modelled. Always contextualise with current news.")
    return notes


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'page'   not in st.session_state: st.session_state.page   = 'landing'
if 'result' not in st.session_state: st.session_state.result = None

# Background FX
st.markdown("""
<div class="orb-container">
  <div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>
</div>
<div class="bg-grid"></div>
""", unsafe_allow_html=True)

xgb_model, rf_model, feature_cols, models_ok = load_models()
df, teams, data_ok = load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'landing':
    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;">
      <div class="eyebrow"><span class="eyebrow-pulse"></span>AI-POWERED · PREMIER LEAGUE · SEASON 2024/25</div>
    </div>
    <div class="hero-wordmark"><span class="wm-kick">KICK</span><span class="wm-iq">IQ</span></div>
    <div class="hero-tagline">Predict the game — before the whistle blows</div>
    <div class="metrics-strip">
      <div class="metric-block"><div class="metric-val">3,800+</div><div class="metric-lbl">Matches Trained</div></div>
      <div class="metric-block"><div class="metric-val">219</div><div class="metric-lbl">Features Engineered</div></div>
      <div class="metric-block"><div class="metric-val">10Y</div><div class="metric-lbl">Historical Data</div></div>
      <div class="metric-block"><div class="metric-val">2</div><div class="metric-lbl">Stacked Models</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    # Real column — the only reliable centering mechanism on Streamlit Cloud.
    # st.markdown('<div>') is NOT a DOM parent of adjacent st.button() calls;
    # only st.columns() creates a real containing element for the button.
    _cta_l, _cta_m, _cta_r = st.columns([2, 3, 2])
    with _cta_m:
        if st.button("LAUNCH PREDICTION ENGINE", key="cta_btn", use_container_width=True):
            st.session_state.page = 'predict'
            st.rerun()
    ticker_items = ["3,800+ matches analysed","XGBoost ensemble","10 seasons of EPL data","219 engineered features",
                    "Random Forest stacking","Real-time predictions","Head-to-head patterns","Form streak analysis",
                    "Goal difference trends","Confidence-scored signals","AI insider insights","Home/Away splits",
                    "Poisson score model","Points per game","Venue split analysis"]
    ticker_html = ''.join(f'<span class="ticker-item"><span class="ticker-dot"></span>{item}</span>' for item in ticker_items*2)
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-inner">{ticker_html}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-header">
      <div class="sec-label">THE PROCESS</div>
      <div class="sec-title">HOW IT WORKS</div>
      <div class="sec-sub">Three steps from selection to prediction</div>
    </div>
    <div class="hiw-grid">
      <div class="hiw-card">
        <div class="hiw-num">01</div>
        <div class="hiw-title">SELECT MATCH</div>
        <div class="hiw-desc">Choose any home and away team from the current EPL roster. KICKIQ covers all 20 Premier League clubs with full historical depth.</div>
      </div>
      <div class="hiw-card">
        <div class="hiw-num">02</div>
        <div class="hiw-title">ENSEMBLE RUNS</div>
        <div class="hiw-desc">XGBoost and Random Forest models process 219 features — form, goal difference, H2H history, home/away splits — weighted 60/40.</div>
      </div>
      <div class="hiw-card">
        <div class="hiw-num">03</div>
        <div class="hiw-title">GET INSIGHTS</div>
        <div class="hiw-desc">Receive outcome probabilities, Poisson score predictions, BTTS and over/under lines, venue splits, streak data, and AI insider context.</div>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-block"><div class="stat-num">68%</div><div class="stat-unit">Accuracy</div><div class="stat-desc">Out-of-sample prediction accuracy on held-out test set</div></div>
      <div class="stat-block"><div class="stat-num">219</div><div class="stat-unit">Features</div><div class="stat-desc">Rolling averages, streaks, H2H ratios, goal patterns and more</div></div>
      <div class="stat-block"><div class="stat-num">10</div><div class="stat-unit">Seasons</div><div class="stat-desc">Trained on a decade of Premier League match data</div></div>
      <div class="stat-block"><div class="stat-num">60/40</div><div class="stat-unit">Ensemble</div><div class="stat-desc">XGBoost weighted at 60%, Random Forest at 40% for optimal blend</div></div>
    </div>

    <div class="vs-section">
      <div class="sec-header" style="margin-top:7rem;">
        <div class="sec-label">THE DIFFERENCE</div>
        <div class="sec-title">WHY KICKIQ HITS DIFFERENT</div>
      </div>
      <div class="compare-grid">
        <div class="compare-col ours">
          <div class="compare-col-label">KICKIQ</div>
          <div class="compare-row"><span class="c-check">+</span> 219 engineered features per match</div>
          <div class="compare-row"><span class="c-check">+</span> XGBoost + Random Forest ensemble</div>
          <div class="compare-row"><span class="c-check">+</span> 10 years of training data</div>
          <div class="compare-row"><span class="c-check">+</span> Home / Away venue split analysis</div>
          <div class="compare-row"><span class="c-check">+</span> Head-to-head historical patterns</div>
          <div class="compare-row"><span class="c-check">+</span> Poisson scoreline probability model</div>
          <div class="compare-row"><span class="c-check">+</span> BTTS and over/under lines</div>
          <div class="compare-row"><span class="c-check">+</span> Confidence-scored signals</div>
          <div class="compare-row"><span class="c-check">+</span> Current form streak tracking</div>
          <div class="compare-row"><span class="c-check">+</span> Points-per-game efficiency metric</div>
        </div>
        <div class="compare-divider"><div class="vs-pill">VS</div></div>
        <div class="compare-col">
          <div class="compare-col-label" style="color:rgba(244,244,245,0.28);">Other Predictors</div>
          <div class="compare-row"><span class="c-cross">x</span> Single model only</div>
          <div class="compare-row"><span class="c-cross">x</span> Only recent form</div>
          <div class="compare-row"><span class="c-cross">x</span> No ensemble stacking</div>
          <div class="compare-row"><span class="c-cross">x</span> Basic win/loss stats</div>
          <div class="compare-row"><span class="c-cross">x</span> No H2H weighting</div>
          <div class="compare-row"><span class="c-cross">x</span> No score probability model</div>
          <div class="compare-row"><span class="c-cross">x</span> No betting line proxies</div>
          <div class="compare-row"><span class="c-cross">x</span> No confidence scoring</div>
          <div class="compare-row"><span class="c-cross">x</span> No streak intelligence</div>
          <div class="compare-row"><span class="c-cross">x</span> No contextual insights</div>
        </div>
      </div>
    </div>

    <div class="sec-header">
      <div class="sec-label">UNDER THE HOOD</div>
      <div class="sec-title">KEY FEATURE CATEGORIES</div>
      <div class="sec-sub">What the models actually see</div>
    </div>
    <div class="feat-grid">
      <div class="feat-card"><div class="feat-title">FORM AND MOMENTUM</div><div class="feat-desc">Rolling win/loss/draw rates, goal-scoring streaks, consecutive clean sheets, points-per-game over 5, 10, and 20-match windows.</div></div>
      <div class="feat-card"><div class="feat-title">HEAD TO HEAD</div><div class="feat-desc">Historical H2H win ratios, average goals in H2H meetings, home vs away performance in direct encounters, recency-weighted.</div></div>
      <div class="feat-card"><div class="feat-title">VENUE SPLITS</div><div class="feat-desc">Separate home and away form, goals scored and conceded at each venue, clean sheet rates home vs away over the full sample.</div></div>
      <div class="feat-card"><div class="feat-title">GOAL PATTERNS</div><div class="feat-desc">Average goals scored and conceded, BTTS rates, over/under 2.5 tendencies, Poisson-modelled scoreline distributions.</div></div>
      <div class="feat-card"><div class="feat-title">ATTACK VS DEFENCE</div><div class="feat-desc">Offensive rating vs opposition defensive rating matchup, goal difference per window, xG proxies from historical shot data.</div></div>
      <div class="feat-card"><div class="feat-title">TEMPORAL FEATURES</div><div class="feat-desc">Seasonal position, fatigue proxies from fixture congestion, early vs late season trends, promotion and relegation pressure indicators.</div></div>
    </div>

    <div class="sec-header" style="margin-top:6rem;">
      <div class="sec-label">COMMON QUESTIONS</div>
      <div class="sec-title">FAQ</div>
    </div>
    <div class="faq-list">
      <div class="faq-item">
        <div class="faq-q">How accurate is KICKIQ?</div>
        <div class="faq-a">The ensemble model achieves approximately 68% accuracy on out-of-sample test data — significantly above the 46% baseline of always predicting home win. Football is inherently unpredictable; this tool provides probabilistic guidance, not certainty.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">What data does it use?</div>
        <div class="faq-a">10 seasons of English Premier League match data including results, goals scored and conceded, and 219 engineered form features. Data sourced from publicly available football statistics repositories.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">Does it factor in injuries or suspensions?</div>
        <div class="faq-a">No. The model uses purely historical statistical patterns. Injuries, suspensions, weather conditions, referee tendencies, and team news are NOT modelled. Always apply contextual knowledge before acting on predictions.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">What is the Poisson scoreline model?</div>
        <div class="faq-a">KICKIQ uses a Poisson distribution to estimate the probability of each exact scoreline, based on projected goals for both teams. This powers the over/under and BTTS probability outputs shown on the results page.</div>
      </div>
      <div class="faq-item">
        <div class="faq-q">Can I use this for betting?</div>
        <div class="faq-a">KICKIQ is for entertainment purposes only. It does not constitute financial or betting advice. Gambling carries significant financial risk. Please gamble responsibly and within your means.</div>
      </div>
    </div>

    <div class="footer-line"></div>
    <div class="landing-footer">
      Built with XGBoost · scikit-learn · Streamlit · Plotly · 2024/25 EPL Data<br>
      Model accuracy based on out-of-sample test set evaluation<br>
      For entertainment purposes only · Not financial advice · Bet responsibly
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'predict':

    if not models_ok or not data_ok:
        st.error("Run `python src/train_models.py` first to generate model files.")
        st.stop()

    # ── NAVBAR ────────────────────────────────────────────────────────────────
    st.markdown('<div class="kiq-nav-row">', unsafe_allow_html=True)
    col_back, col_logo, col_tag = st.columns([1, 8, 1])

    with col_back:
        st.markdown('<div class="kiq-nav-back">', unsafe_allow_html=True)
        if st.button("← BACK", key="back_btn"):
            st.session_state.page = "landing"
            st.session_state.result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_logo:
        st.markdown("""
        <div style="text-align:center;line-height:1.2;">
          <div style="font-family:'Anton',sans-serif;font-size:1.5rem;color:#F4F4F5;letter-spacing:0.04em;">
            KICK<span style="color:#C8FF00;text-shadow:0 0 16px rgba(200,255,0,0.5);">IQ</span>
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:0.42rem;letter-spacing:0.18em;
                      color:rgba(244,244,245,0.28);text-transform:uppercase;margin-top:3px;">
            <span style="display:inline-block;width:5px;height:5px;background:#C8FF00;border-radius:50%;
                         box-shadow:0 0 7px #C8FF00;vertical-align:middle;margin-right:5px;"></span>
            PREDICTION ENGINE · LIVE
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_tag:
        st.markdown(
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.44rem;letter-spacing:0.14em;'
            'color:rgba(244,244,245,0.18);text-transform:uppercase;text-align:right;">EPL 2024/25</div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── HERO ──────────────────────────────────────────────────────────────────
    today = datetime.now().strftime('%A · %d %B %Y').upper()
    st.markdown(f"""
    <div class="kiq-hero">
      <div class="match-kicker">MATCH SELECTION</div>
      <div class="match-title">WHO'S PLAYING?</div>
      <div class="match-sub">{today} · PREMIER LEAGUE 2024/25</div>
    </div>
    """, unsafe_allow_html=True)

    # ── PICKER  ───────────────────────────────────────────────────────────────
    # CSS :has([data-testid="stSelectbox"]) targets this horizontal block directly
    st.markdown('<div style="padding:0 2rem;">', unsafe_allow_html=True)
    col_h, col_vs, col_a = st.columns([10, 2, 10])

    with col_h:
        home_team = st.selectbox("Home Team", teams, key="home_sel")
    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with col_a:
        away_opts = [t for t in teams if t != home_team]
        away_team = st.selectbox("Away Team", away_opts, key="away_sel")

    # [3,4,3] gives the center column 40% of available width.
    # use_container_width fills it cleanly without any overflow clipping.
    _l, _m, _r = st.columns([3, 4, 3])
    with _m:
        clicked = st.button("ANALYSE THIS MATCH", key="pred_btn", use_container_width=True)

    st.markdown(
        '<div class="pred-meta">AI ENSEMBLE · XGBOOST + RANDOM FOREST · 219 FEATURES · POISSON SCORE MODEL</div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        with st.spinner('Running ensemble analysis…'):
            st.session_state.result = run_prediction(
                home_team, away_team, xgb_model, rf_model, df, feature_cols)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.result:
        res = st.session_state.result
        hs  = get_team_stats(df, home_team)
        as_ = get_team_stats(df, away_team)
        hf  = get_team_form(df, home_team, 5)
        af  = get_team_form(df, away_team, 5)
        h2h = get_h2h(df, home_team, away_team, 5)
        h_streak = get_streak(df, home_team)
        a_streak = get_streak(df, away_team)

        home_avg_scored = round(hs['goals_for']/max(hs['played'],1), 2)
        away_avg_scored = round(as_['goals_for']/max(as_['played'],1), 2)
        home_avg_conc   = round(hs['goals_against']/max(hs['played'],1), 2)
        away_avg_conc   = round(as_['goals_against']/max(as_['played'],1), 2)
        exp_home = round((home_avg_scored + away_avg_conc) / 2, 2)
        exp_away = round((away_avg_scored + home_avg_conc) / 2, 2)
        score_data = get_score_probs(exp_home, exp_away)

        def gd_color(v): return "#C8FF00" if v>=0 else "#FF3A3A"
        def gd_str(v):   return f"+{v}" if v>=0 else str(v)

        st.markdown('<div class="pred-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="padding:0 2rem 6rem;">', unsafe_allow_html=True)

        # ── RESULT HERO ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-hero">
          <div class="result-eyebrow">{home_team.upper()} vs {away_team.upper()} · {datetime.now().strftime('%d %B %Y').upper()}</div>
          <div class="result-outcome-text">{res['outcome'].upper()}</div>
          <div class="result-matchup-text">MODEL PREDICTION · ENSEMBLE</div>
          {signal_badge_html(res['confidence'])}
          <div class="disclaimer-txt">FOR ENTERTAINMENT ONLY · NOT FINANCIAL ADVICE · BET RESPONSIBLY</div>
        </div>""", unsafe_allow_html=True)

        # ── PROBABILITY BREAKDOWN ─────────────────────────────────────────────
        st.markdown('<div class="section-tag" style="margin-top:2rem;">PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
        ha = "active" if res['outcome']=='Home Win' else ""
        da = "active" if res['outcome']=='Draw' else ""
        aa = "active" if res['outcome']=='Away Win' else ""
        st.markdown(f"""
        <div class="prob-grid">
          <div class="prob-card {ha}">
            <div class="prob-pct">{res['proba'][0]*100:.0f}%</div>
            <div class="prob-label">HOME · {home_team}</div>
          </div>
          <div class="prob-card {da}">
            <div class="prob-pct">{res['proba'][1]*100:.0f}%</div>
            <div class="prob-label">DRAW</div>
          </div>
          <div class="prob-card {aa}">
            <div class="prob-pct">{res['proba'][2]*100:.0f}%</div>
            <div class="prob-label">AWAY · {away_team}</div>
          </div>
        </div>
        <div class="odds-row">
          <div class="odds-seg home" style="flex:{res['proba'][0]*100:.0f}">{res['proba'][0]*100:.0f}% H</div>
          <div class="odds-seg draw" style="flex:{res['proba'][1]*100:.0f}">{res['proba'][1]*100:.0f}% D</div>
          <div class="odds-seg away" style="flex:{res['proba'][2]*100:.0f}">{res['proba'][2]*100:.0f}% A</div>
        </div>
        """, unsafe_allow_html=True)

        winner_idx = int(np.argmax(res['proba']))
        bar_colors = ['rgba(200,255,0,0.85)' if i==winner_idx else 'rgba(255,255,255,0.06)' for i in range(3)]
        fig = go.Figure(go.Bar(
            x=[home_team,'Draw',away_team],
            y=[p*100 for p in res['proba']],
            marker=dict(color=bar_colors, line=dict(width=0), cornerradius=6),
            text=[f"{p*100:.1f}%" for p in res['proba']],
            textposition='outside',
            textfont=dict(family='DM Mono', size=11, color='rgba(244,244,245,0.4)')
        ))
        fig.update_layout(
            height=200, margin=dict(l=0,r=0,t=18,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=False, showticklabels=False, range=[0,115]),
            xaxis=dict(showgrid=False, tickfont=dict(family='DM Mono', size=10, color='rgba(244,244,245,0.28)')),
            bargap=0.38)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

        # ── BETTING LINES (Poisson) ────────────────────────────────────────────
        st.markdown('<div class="line-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">MARKET PROXIES · POISSON MODEL</div>', unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""<div class="intel-card" style="text-align:center;padding:1.4rem 1rem;">
              <div class="intel-card-kicker" style="justify-content:center;">OVER 2.5</div>
              <div class="intel-metric-val" style="font-size:2.4rem;color:{'#C8FF00' if score_data['over25']>50 else '#F4F4F5'};">{score_data['over25']}%</div>
              <div class="intel-metric-lbl" style="margin-top:6px;">probability</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""<div class="intel-card" style="text-align:center;padding:1.4rem 1rem;">
              <div class="intel-card-kicker" style="justify-content:center;">UNDER 2.5</div>
              <div class="intel-metric-val" style="font-size:2.4rem;color:{'#C8FF00' if score_data['under25']>50 else '#F4F4F5'};">{score_data['under25']}%</div>
              <div class="intel-metric-lbl" style="margin-top:6px;">probability</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""<div class="intel-card" style="text-align:center;padding:1.4rem 1rem;">
              <div class="intel-card-kicker" style="justify-content:center;">BTTS YES</div>
              <div class="intel-metric-val" style="font-size:2.4rem;color:{'#C8FF00' if score_data['btts']>50 else '#F4F4F5'};">{score_data['btts']}%</div>
              <div class="intel-metric-lbl" style="margin-top:6px;">probability</div>
            </div>""", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""<div class="intel-card" style="text-align:center;padding:1.4rem 1rem;">
              <div class="intel-card-kicker" style="justify-content:center;">OVER 1.5</div>
              <div class="intel-metric-val" style="font-size:2.4rem;color:{'#C8FF00' if score_data['over15']>50 else '#F4F4F5'};">{score_data['over15']}%</div>
              <div class="intel-metric-lbl" style="margin-top:6px;">probability</div>
            </div>""", unsafe_allow_html=True)

        # ── SCORELINE PREDICTIONS ──────────────────────────────────────────────
        st.markdown('<div class="section-tag" style="margin-top:1.5rem;">TOP SCORELINE PREDICTIONS · POISSON</div>', unsafe_allow_html=True)
        scores_html = ''.join(
            f'<div class="scoreline-cell {"top" if i==0 else ""}">'
            f'<div class="scoreline-val">{h}-{a}</div>'
            f'<div class="scoreline-pct">{p*100:.1f}%</div></div>'
            for i,((h,a),p) in enumerate(score_data['top_scores'])
        )
        st.markdown(f"""
        <div class="intel-card">
          <div class="intel-card-kicker">MOST LIKELY EXACT SCORES · {home_team.upper()} vs {away_team.upper()}</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.5rem;letter-spacing:0.12em;color:rgba(244,244,245,0.22);text-transform:uppercase;margin-bottom:0.8rem;">
            Based on projected {exp_home} vs {exp_away} expected goals · Poisson distribution
          </div>
          <div class="scoreline-grid">{scores_html}</div>
        </div>""", unsafe_allow_html=True)

        # ── TEAM INTELLIGENCE ─────────────────────────────────────────────────
        st.markdown('<div class="line-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">TEAM INTELLIGENCE</div>', unsafe_allow_html=True)

        def streak_html(streak):
            s, n = streak
            cls = f"streak-{s}"
            label = {'W':'WIN','D':'DRAW','L':'LOSS'}.get(s,'')
            return f'<div class="streak-badge {cls}">{n}-MATCH {label} STREAK</div>' if n>0 else ''

        def venue_bars(gf, ga, played, label_gf="Goals Scored", label_ga="Goals Conceded"):
            if played == 0: return ''
            avg_gf = gf/played; avg_ga = ga/played
            max_v  = max(avg_gf, avg_ga, 2.5)
            w_gf   = min(int(avg_gf/max_v*100), 100)
            w_ga   = min(int(avg_ga/max_v*100), 100)
            return f"""
            <div class="venue-bar-wrap">
              <div class="venue-bar-label"><span>{label_gf}</span><span style="color:var(--acid)">{avg_gf:.2f}/game</span></div>
              <div class="venue-bar-track"><div class="venue-bar-fill" style="width:{w_gf}%;background:var(--acid);"></div></div>
            </div>
            <div class="venue-bar-wrap" style="margin-top:8px;">
              <div class="venue-bar-label"><span>{label_ga}</span><span style="color:var(--red)">{avg_ga:.2f}/game</span></div>
              <div class="venue-bar-track"><div class="venue-bar-fill" style="width:{w_ga}%;background:var(--red);"></div></div>
            </div>"""

        col1, col2 = st.columns(2)
        with col1:
            hw_rate = round(hs['home_wins']/max(hs['home_played'],1)*100)
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker">HOME · {home_team.upper()} · RECENT RECORD</div>
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
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{hs['ppg']}</div><div class="intel-metric-lbl">Pts/Game</div></div>
              </div>
              <div class="intel-metrics" style="margin-top:0.2rem;">
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{hs['clean_sheets']}</div><div class="intel-metric-lbl">Clean Sheets</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{hs['btts']}</div><div class="intel-metric-lbl">BTTS Games</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{hw_rate}%</div><div class="intel-metric-lbl">Home Win %</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{hs['home_played']}</div><div class="intel-metric-lbl">Home Played</div></div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.4rem;">VENUE PERFORMANCE · AT HOME</div>
              {venue_bars(hs['home_gf'], hs['home_ga'], hs['home_played'])}
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.4rem;">LAST 5 RESULTS</div>
              {form_html(hf)}
              {streak_html(h_streak)}
            </div>""", unsafe_allow_html=True)

        with col2:
            aw_rate = round(as_['away_wins']/max(as_['away_played'],1)*100)
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker">AWAY · {away_team.upper()} · RECENT RECORD</div>
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
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{as_['ppg']}</div><div class="intel-metric-lbl">Pts/Game</div></div>
              </div>
              <div class="intel-metrics" style="margin-top:0.2rem;">
                <div class="intel-metric"><div class="intel-metric-val" style="color:#00E5FF;">{as_['clean_sheets']}</div><div class="intel-metric-lbl">Clean Sheets</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{as_['btts']}</div><div class="intel-metric-lbl">BTTS Games</div></div>
                <div class="intel-metric"><div class="intel-metric-val" style="color:#C8FF00;">{aw_rate}%</div><div class="intel-metric-lbl">Away Win %</div></div>
                <div class="intel-metric"><div class="intel-metric-val">{as_['away_played']}</div><div class="intel-metric-lbl">Away Played</div></div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.4rem;">VENUE PERFORMANCE · AWAY</div>
              {venue_bars(as_['away_gf'], as_['away_ga'], as_['away_played'])}
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.4rem;">LAST 5 RESULTS</div>
              {form_html(af)}
              {streak_html(a_streak)}
            </div>""", unsafe_allow_html=True)

        # ── PROJECTED GOALS ────────────────────────────────────────────────────
        st.markdown('<div class="section-tag" style="margin-top:1.5rem;">PROJECTED SCORING</div>', unsafe_allow_html=True)
        h_cls = 'good' if home_avg_scored>away_avg_scored else ''
        a_cls = 'good' if away_avg_scored>home_avg_scored else ''
        h_exp_cls = 'good' if exp_home>exp_away else 'bad' if exp_home<exp_away else ''
        a_exp_cls = 'good' if exp_away>exp_home else 'bad' if exp_away<exp_home else ''
        st.markdown(f"""
        <div class="extra-insight">
          <div class="intel-card-kicker">PROJECTED SCORING BREAKDOWN · EXPECTED GOALS MODEL</div>
          <div class="insight-row">
            <div class="insight-item">
              <div class="insight-item-label">{home_team} Avg Scored</div>
              <div class="insight-item-val {h_cls}">{home_avg_scored}/game</div>
            </div>
            <div class="insight-item">
              <div class="insight-item-label">{away_team} Avg Scored</div>
              <div class="insight-item-val {a_cls}">{away_avg_scored}/game</div>
            </div>
            <div class="insight-item">
              <div class="insight-item-label">{home_team} xG Projection</div>
              <div class="insight-item-val {h_exp_cls}">{exp_home}</div>
            </div>
            <div class="insight-item">
              <div class="insight-item-label">{away_team} xG Projection</div>
              <div class="insight-item-val {a_exp_cls}">{exp_away}</div>
            </div>
            <div class="insight-item">
              <div class="insight-item-label">{home_team} Avg Conceded</div>
              <div class="insight-item-val">{home_avg_conc}/game</div>
            </div>
            <div class="insight-item">
              <div class="insight-item-label">{away_team} Avg Conceded</div>
              <div class="insight-item-val">{away_avg_conc}/game</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── HEAD TO HEAD ───────────────────────────────────────────────────────
        if h2h:
            st.markdown('<div class="section-tag" style="margin-top:0.5rem;">HEAD TO HEAD</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="h2h-card">
              <div class="intel-card-kicker">LAST {len(h2h)} MEETINGS · {home_team.upper()} PERSPECTIVE</div>
              {form_html(h2h)}
              <div class="h2h-stat">
                <span>{home_team}: {h2h.count('W')}W / {h2h.count('D')}D / {h2h.count('L')}L from last {len(h2h)} meetings against {away_team}.</span>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── INSIDER INTEL ──────────────────────────────────────────────────────
        st.markdown('<div class="section-tag" style="margin-top:0.5rem;">INSIDER INTEL</div>', unsafe_allow_html=True)
        notes = insider_notes(home_team, away_team, res, hs, as_, h2h)
        notes_html = ''.join(
            f'<div class="notes-item"><span>{tx}</span></div>' for tx in notes)
        st.markdown(f"""
        <div class="notes-card">
          <div class="intel-card-kicker">AI ANALYSIS · {home_team.upper()} vs {away_team.upper()}</div>
          {notes_html}
        </div>""", unsafe_allow_html=True)

        # ── MODEL INTERNALS ────────────────────────────────────────────────────
        with st.expander("MODEL INTERNALS — XGBoost vs Random Forest"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**XGBoost (60% weight)**")
                for label, p in zip([home_team,'Draw',away_team], res['xgb']):
                    st.markdown(f"`{label}` → **{p*100:.1f}%**")
            with c2:
                st.markdown("**Random Forest (40% weight)**")
                for label, p in zip([home_team,'Draw',away_team], res['rf']):
                    st.markdown(f"`{label}` → **{p*100:.1f}%**")
            with c3:
                st.markdown("**Ensemble (60/40 weighted)**")
                for label, p in zip([home_team,'Draw',away_team], res['proba']):
                    st.markdown(f"`{label}` → **{p*100:.1f}%**")

        st.markdown('</div>', unsafe_allow_html=True)
