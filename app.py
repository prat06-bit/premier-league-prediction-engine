import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime

st.set_page_config(page_title="KICKIQ · EPL Predictor", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=DM+Mono:wght@300;400;500&family=Clash+Display:wght@400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700;800;900&family=Saira+Condensed:wght@100;200;300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --acid: #C8FF00;
  --acid-dim: rgba(200,255,0,0.15);
  --acid-glow: rgba(200,255,0,0.35);
  --void: #050608;
  --surface: #0A0C0F;
  --surface2: #0F1115;
  --border: rgba(255,255,255,0.06);
  --border-bright: rgba(200,255,0,0.25);
  --text: #F4F4F5;
  --muted: rgba(244,244,245,0.4);
  --faint: rgba(244,244,245,0.12);
  --red: #FF3A3A;
  --amber: #FFB800;
  --cyan: #00E5FF;
}

html, body, .stApp { background: var(--void) !important; color: var(--text); font-family: 'Rajdhani', sans-serif; overflow-x: hidden; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--acid); border-radius: 2px; }


body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
  mix-blend-mode: overlay;
}

body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.03) 2px,
    rgba(0,0,0,0.03) 4px
  );
  pointer-events: none;
  z-index: 9998;
}

.orb-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;      /* CRITICAL */
  pointer-events: none;
  z-index: -2;        /* send fully behind */
  overflow: hidden;
}

.orb-1 { width: 700px; height: 700px; top: -20%; left: -15%; background: radial-gradient(circle, rgba(200,255,0,0.09) 0%, transparent 70%); --dur: 18s; }
.orb-2 { width: 500px; height: 500px; bottom: -15%; right: -10%; background: radial-gradient(circle, rgba(0,229,255,0.07) 0%, transparent 70%); --dur: 23s; animation-delay: -7s; animation-direction: reverse; }
.orb-3 { width: 400px; height: 400px; top: 40%; left: 50%; background: radial-gradient(circle, rgba(200,255,0,0.04) 0%, transparent 70%); --dur: 30s; animation-delay: -12s; }

@keyframes orbFloat {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(30px, -40px) scale(1.05); }
  50% { transform: translate(-20px, 30px) scale(0.95); }
  75% { transform: translate(40px, 20px) scale(1.03); }
}

.bg-grid {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;        /* CRITICAL FIX */
  z-index: -1;          /* send behind everything */
  pointer-events: none;

  background-image:
    linear-gradient(rgba(200,255,0,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(200,255,0,0.025) 1px, transparent 1px);

  background-size: 80px 80px;

  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}

.landing-wrap {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;   
  padding-top: 10vh;              
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: linear-gradient(135deg, rgba(200,255,0,0.08), rgba(200,255,0,0.03));
  border: 1px solid rgba(200,255,0,0.2);
  border-radius: 100px;
  padding: 7px 18px 7px 12px;
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  color: var(--acid);
  text-transform: uppercase;
  margin-bottom: 2.5rem;
  animation: slideDown 0.7s cubic-bezier(0.16,1,0.3,1) both;
}
.eyebrow-pulse {
  width: 7px; height: 7px;
  background: var(--acid);
  border-radius: 50%;
  animation: pulse 1.8s ease-in-out infinite;
  box-shadow: 0 0 8px var(--acid);
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.3;transform:scale(0.7);} }

.hero-wordmark {
  font-family: 'Anton', sans-serif;
  font-size: clamp(7rem, 18vw, 15rem);
  line-height: 0.85;
  letter-spacing: -0.02em;
  text-align: center;
  position: relative;
  z-index: 1;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.1s both;
}
.wm-kick {
  color: var(--text);
  display: inline;
  position: relative;
}
.wm-iq {
  color: var(--acid);
  display: inline;
  text-shadow: 0 0 60px rgba(200,255,0,0.5), 0 0 120px rgba(200,255,0,0.2);
  animation: acidFlicker 6s ease-in-out infinite 2s;
}
@keyframes acidFlicker {
  0%,90%,100% { text-shadow: 0 0 60px rgba(200,255,0,0.5), 0 0 120px rgba(200,255,0,0.2); }
  92% { text-shadow: 0 0 10px rgba(200,255,0,0.2); opacity: 0.9; }
  94% { text-shadow: 0 0 60px rgba(200,255,0,0.5), 0 0 120px rgba(200,255,0,0.2); }
  96% { text-shadow: 0 0 10px rgba(200,255,0,0.2); opacity: 0.85; }
  98% { text-shadow: 0 0 80px rgba(200,255,0,0.6), 0 0 140px rgba(200,255,0,0.25); }
}

.hero-tagline {
  font-family: 'Saira Condensed', sans-serif;
  font-weight: 300;
  font-size: clamp(0.95rem, 2vw, 1.2rem);
  letter-spacing: 0.38em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: center;
  margin: 1.5rem 0 0;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.3s both;
}

.hero-desc {
  font-family: 'Rajdhani', sans-serif;
  font-weight: 400;
  font-size: clamp(1rem, 1.8vw, 1.15rem);
  color: rgba(244,244,245,0.5);
  text-align: center;
  max-width: 480px;
  line-height: 1.7;
  margin: 1.8rem auto 0;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.4s both;
}

@keyframes heroReveal {
  from { opacity: 0; transform: translateY(24px) skewY(1deg); }
  to { opacity: 1; transform: translateY(0) skewY(0); }
}
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-12px); }
  to { opacity: 1; transform: translateY(0); }
}

.metrics-strip {
  display: flex;
  gap: 0;
  align-items: stretch;
  margin: 4.5rem 0 3.5rem;
  position: relative;
  z-index: 1;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.5s both;
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  background: var(--surface);
}
.metric-block {
  flex: 1;
  padding: 2rem 2.5rem;
  position: relative;
  border-right: 1px solid var(--border);
  transition: background 0.3s;
}
.metric-block:last-child { border-right: none; }
.metric-block:hover { background: rgba(200,255,0,0.03); }
.metric-block::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--acid), transparent);
  transform: scaleX(0);
  transition: transform 0.4s ease;
}
.metric-block:hover::after { transform: scaleX(1); }
.metric-val {
  font-family: 'Anton', sans-serif;
  font-size: 3.2rem;
  line-height: 1;
  color: var(--acid);
  letter-spacing: -0.01em;
}
.metric-lbl {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 6px;
}

.cta-section {
  position: relative;
  z-index: 1;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.65s both;
}
.stButton > button {
  background: var(--acid) !important;
  color: #050608 !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Anton', sans-serif !important;
  font-size: 1.35rem !important;
  letter-spacing: 0.12em !important;
  padding: 1.1rem 2.5rem !important;
  width: 100% !important;
  transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 50%);
  opacity: 0;
  transition: opacity 0.2s;
}
.stButton > button:hover {
  transform: translateY(-3px) scale(1.01) !important;
  box-shadow: 0 0 40px rgba(200,255,0,0.5), 0 8px 32px rgba(200,255,0,0.2) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

.vs-section {
  width: 100%;
  max-width: 860px;
  margin: 5rem auto 0;
  position: relative;
  z-index: 1;
  animation: heroReveal 0.9s cubic-bezier(0.16,1,0.3,1) 0.8s both;
}
.vs-heading {
  font-family: 'Saira Condensed', sans-serif;
  font-weight: 200;
  font-size: 0.75rem;
  letter-spacing: 0.35em;
  color: var(--muted);
  text-transform: uppercase;
  text-align: center;
  margin-bottom: 2rem;
}
.compare-grid {
  display: grid;
  grid-template-columns: 1fr 40px 1fr;
  gap: 1px;
  background: var(--border);
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.compare-col {
  background: var(--surface);
  padding: 1.8rem;
}
.compare-col.ours {
  background: linear-gradient(135deg, rgba(200,255,0,0.06), rgba(200,255,0,0.02));
  border-left: 2px solid rgba(200,255,0,0.3);
}
.compare-col-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 1.4rem;
  color: var(--muted);
}
.compare-col.ours .compare-col-label { color: var(--acid); }
.compare-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.95rem;
  font-weight: 500;
  color: rgba(244,244,245,0.65);
}
.compare-row:last-child { border-bottom: none; }
.compare-col.ours .compare-row { color: rgba(244,244,245,0.8); }
.c-check { color: var(--acid); font-size: 0.85rem; }
.c-cross { color: rgba(244,244,245,0.2); font-size: 0.85rem; }
.compare-divider {
  background: var(--surface);
  display: flex;
  align-items: center;
  justify-content: center;
}
.vs-pill {
  font-family: 'Anton', sans-serif;
  font-size: 1.1rem;
  color: var(--border);
  writing-mode: vertical-rl;
  letter-spacing: 0.4em;
}

.landing-footer {
  font-family: 'DM Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  color: rgba(244,244,245,0.18);
  text-align: center;
  text-transform: uppercase;
  margin-top: 4rem;
  padding-bottom: 3rem;
  line-height: 2;
  position: relative;
  z-index: 1;
}

.pred-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.6rem 3rem;
  border-bottom: 1px solid var(--border);
  position: relative;
  z-index: 1;
  background: rgba(5,6,8,0.8);
  backdrop-filter: blur(20px);
  position: sticky;
  top: 0;
}
.pred-logo-text {
  font-family: 'Anton', sans-serif;
  font-size: 1.8rem;
  letter-spacing: 0.04em;
  color: var(--text);
  line-height: 1;
}
.pred-logo-text .iq { color: var(--acid); }
.pred-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 3px;
}
.live-dot {
  width: 6px; height: 6px;
  background: var(--acid);
  border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 8px var(--acid);
  animation: pulse 1.5s ease-in-out infinite;
  margin-right: 6px;
}

.selector-zone {
  padding: 3.5rem 3rem 1.5rem;
  position: relative;
  z-index: 1;
}
.section-kicker {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.25em;
  color: var(--acid);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-kicker::before {
  content: '';
  width: 24px; height: 1px;
  background: var(--acid);
}
.section-title {
  font-family: 'Anton', sans-serif;
  font-size: clamp(2.5rem, 5vw, 4rem);
  letter-spacing: 0.01em;
  line-height: 1;
  margin-bottom: 2.5rem;
  color: var(--text);
}

.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  transition: border-color 0.2s !important;
}
.stSelectbox > div > div:hover {
  border-color: rgba(200,255,0,0.4) !important;
  background: rgba(200,255,0,0.02) !important;
}
.stSelectbox label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

.result-hero {
  background: linear-gradient(135deg, rgba(200,255,0,0.07), rgba(200,255,0,0.02));
  border: 1px solid rgba(200,255,0,0.2);
  border-radius: 20px;
  padding: 3rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  margin: 1.5rem 0;
}
.result-hero::before {
  content: '';
  position: absolute;
  top: 0; left: -100%; right: -100%;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--acid), transparent);
  animation: scanline 3s linear infinite;
}
@keyframes scanline { from{left:-100%;} to{left:100%;} }

.result-hero::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 60% 60% at 50% 0%, rgba(200,255,0,0.08), transparent 70%);
  pointer-events: none;
}

.result-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.result-outcome-text {
  font-family: 'Anton', sans-serif;
  font-size: clamp(3.5rem, 8vw, 6rem);
  line-height: 1;
  color: var(--acid);
  text-shadow: 0 0 60px rgba(200,255,0,0.4);
  letter-spacing: 0.02em;
  position: relative;
  z-index: 1;
  animation: outcomePop 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes outcomePop {
  from { opacity: 0; transform: scale(0.8) translateY(10px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
.result-matchup-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 0.5rem;
  position: relative;
  z-index: 1;
}

.signal-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 20px;
  border-radius: 100px;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  font-weight: 500;
  margin: 1.2rem 0 0.4rem;
  position: relative;
  z-index: 1;
}
.signal-badge.s-strong { background: rgba(200,255,0,0.1); border: 1px solid rgba(200,255,0,0.35); color: var(--acid); }
.signal-badge.s-mod { background: rgba(255,184,0,0.1); border: 1px solid rgba(255,184,0,0.3); color: var(--amber); }
.signal-badge.s-weak { background: rgba(255,100,0,0.1); border: 1px solid rgba(255,100,0,0.3); color: #FF6400; }
.signal-badge.s-none { background: rgba(255,58,58,0.1); border: 1px solid rgba(255,58,58,0.3); color: var(--red); }

.disclaimer-txt {
  font-family: 'DM Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.1em;
  color: rgba(244,244,245,0.2);
  text-transform: uppercase;
  margin-top: 8px;
  position: relative;
  z-index: 1;
}

.prob-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}
.prob-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.8rem 1.5rem;
  text-align: center;
  transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
  position: relative;
  overflow: hidden;
}
.prob-card:hover { transform: translateY(-3px); border-color: rgba(200,255,0,0.2); }
.prob-card.active {
  background: linear-gradient(135deg, rgba(200,255,0,0.1), rgba(200,255,0,0.04));
  border-color: rgba(200,255,0,0.4);
  box-shadow: 0 0 30px rgba(200,255,0,0.08), inset 0 0 30px rgba(200,255,0,0.03);
}
.prob-card.active::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--acid), transparent);
}
.prob-pct {
  font-family: 'Anton', sans-serif;
  font-size: 3.2rem;
  line-height: 1;
  color: var(--text);
  transition: color 0.3s;
}
.prob-card.active .prob-pct { color: var(--acid); }
.prob-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.14em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 7px;
}

.intel-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.intel-card:hover { border-color: rgba(200,255,0,0.15); }
.intel-card-kicker {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  color: var(--acid);
  text-transform: uppercase;
  margin-bottom: 1.4rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.intel-metrics {
  display: flex;
  gap: 0.7rem;
  margin-bottom: 0.7rem;
  flex-wrap: wrap;
}
.intel-metric {
  flex: 1;
  min-width: 70px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 0.8rem;
}
.intel-metric-val {
  font-family: 'Anton', sans-serif;
  font-size: 2rem;
  line-height: 1;
  color: var(--text);
}
.intel-metric-lbl {
  font-family: 'DM Mono', monospace;
  font-size: 0.56rem;
  letter-spacing: 0.1em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 4px;
}

.form-row { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 0.5rem; }
.fp {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px; height: 30px;
  border-radius: 7px;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  position: relative;
  overflow: hidden;
}
.fp::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
}
.fp-W { background: rgba(200,255,0,0.18); color: var(--acid); border: 1px solid rgba(200,255,0,0.3); }
.fp-D { background: rgba(255,184,0,0.15); color: var(--amber); border: 1px solid rgba(255,184,0,0.25); }
.fp-L { background: rgba(255,58,58,0.12); color: var(--red); border: 1px solid rgba(255,58,58,0.2); }

.h2h-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  margin: 1.5rem 0;
}
.h2h-stat {
  font-family: 'Rajdhani', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  color: rgba(244,244,245,0.6);
  margin-top: 1rem;
  line-height: 1.6;
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.notes-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  margin: 1.5rem 0;
}
.notes-item {
  display: flex;
  gap: 12px;
  padding: 11px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-family: 'Rajdhani', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  color: rgba(244,244,245,0.7);
  line-height: 1.55;
}
.notes-item:last-child { border-bottom: none; }
.note-icon { flex-shrink: 0; font-size: 1.05rem; margin-top: 1px; }

.streamlit-expanderHeader {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.15em !important;
  color: var(--muted) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

.back-btn-wrap .stButton > button {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em !important;
  padding: 0.5rem 1.2rem !important;
  width: auto !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}
.back-btn-wrap .stButton > button:hover {
  border-color: rgba(200,255,0,0.3) !important;
  color: var(--acid) !important;
  transform: none !important;
  box-shadow: none !important;
}

.stSpinner > div { border-top-color: var(--acid) !important; }

.section-tag {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  color: var(--acid);
  text-transform: uppercase;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-tag::before {
  content: '';
  width: 20px; height: 1px;
  background: var(--acid);
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .hero-wordmark { font-size: 5rem; }
  .metrics-strip { flex-direction: column; }
  .metric-block { border-right: none; border-bottom: 1px solid var(--border); }
  .metric-block:last-child { border-bottom: none; }
  .compare-grid { grid-template-columns: 1fr; }
  .compare-divider { display: none; }
  .selector-zone, .pred-topbar, .results-pad { padding-left: 1.5rem !important; padding-right: 1.5rem !important; }
  .prob-grid { grid-template-columns: 1fr 1fr; }
  .prob-grid .prob-card:last-child { grid-column: 1/-1; }
}

.line-sep {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(200,255,0,0.3), transparent);
  margin: 3rem 0;
  position: relative;
  z-index: 1;
  animation: linePulse 3s ease-in-out infinite;
}
@keyframes linePulse {
  0%,100% { opacity: 0.4; }
  50% { opacity: 1; }
}

.pred-topbar {
  transition: backdrop-filter .4s ease, background .4s ease, border-color .4s ease;
  background: rgba(5,6,8,0.55) !important;
  backdrop-filter: blur(26px) saturate(140%);
  border-bottom: 1px solid rgba(200,255,0,0.08);
}

.pred-topbar:hover {
  background: rgba(5,6,8,0.75) !important;
  border-bottom: 1px solid rgba(200,255,0,0.25);
}

.stSelectbox:focus-within div[data-baseweb="select"] {
  box-shadow: 0 0 0 1px rgba(200,255,0,0.6),
              0 0 22px rgba(200,255,0,0.15);
  border-color: rgba(200,255,0,0.5) !important;
  transition: all .2s ease;
}

@keyframes ctaPulse {
  0%   { box-shadow: 0 0 0 rgba(200,255,0,0.0); }
  50%  { box-shadow: 0 0 38px rgba(200,255,0,0.35); }
  100% { box-shadow: 0 0 0 rgba(200,255,0,0.0); }
}

.stButton > button {
  animation: ctaPulse 4s ease-in-out infinite;
}

.result-hero {
  animation: resultSlide .6s cubic-bezier(0.16,1,0.3,1);
}

@keyframes resultSlide {
  from { opacity:0; transform:translateY(35px); }
  to   { opacity:1; transform:translateY(0); }
}
            
.result-outcome-text {
  position: relative;
  overflow: hidden;
}

.result-outcome-text::after {
  content:'';
  position:absolute;
  top:0;
  left:-120%;
  width:120%;
  height:100%;
  background:linear-gradient(
    90deg,
    transparent,
    rgba(255,255,255,0.25),
    transparent
  );
  animation: broadcastSweep 1.2s ease forwards;
}

@keyframes broadcastSweep {
  to { left:120%; }
}


.result-outcome-text {
  animation: cinematicReveal .7s cubic-bezier(0.16,1,0.3,1);
}

@keyframes cinematicReveal {
  0%   { opacity:0; transform:translateY(40px) scale(.95); filter:blur(6px);}
  60%  { opacity:1; filter:blur(0);}
  100% { transform:translateY(0) scale(1);}
}

.result-hero {
  transition: background 0.6s ease, box-shadow .6s ease;
}

.result-hero:hover {
  box-shadow: 0 0 60px rgba(200,255,0,0.25);
}

/* Probability hover glow */
.prob-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow:
    0 0 25px rgba(200,255,0,0.15),
    inset 0 0 25px rgba(200,255,0,0.05);
}

</style>
""", unsafe_allow_html=True)

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
        return {'played':0,'wins':0,'draws':0,'losses':0,'goals_for':0,'goals_against':0,'gd':0,'win_rate':0}
    wins   = (((all_m['venue']=='home')&(all_m['FTR']=='H'))|((all_m['venue']=='away')&(all_m['FTR']=='A'))).sum()
    draws  = (all_m['FTR']=='D').sum()
    losses = len(all_m)-wins-draws
    gf = int(np.where(all_m['venue']=='home', all_m['FTHG'], all_m['FTAG']).sum())
    ga = int(np.where(all_m['venue']=='home', all_m['FTAG'], all_m['FTHG']).sum())
    return {'played':len(all_m),'wins':int(wins),'draws':int(draws),'losses':int(losses),
            'goals_for':gf,'goals_against':ga,'gd':gf-ga,'win_rate':round(wins/len(all_m)*100)}

def get_h2h(df, home, away, n=5):
    h2h = df[((df['HomeTeam']==home)&(df['AwayTeam']==away))|((df['HomeTeam']==away)&(df['AwayTeam']==home))]
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
    feats = pd.DataFrame(index=[0])
    for col in fc:
        if '_home' in col: feats[col] = home_row.get(col, 0)
        elif '_away' in col: feats[col] = away_row.get(col, 0)
        else: feats[col] = 0
    for col in fc:
        if col not in feats.columns: feats[col] = 0
    feats = feats[fc].fillna(0)
    xp = xgb.predict_proba(feats)[0]
    rp = rf.predict_proba(feats)[0]
    ens = 0.6*xp + 0.4*rp
    return {'proba': ens, 'outcome': ['Home Win','Draw','Away Win'][np.argmax(ens)],
            'confidence': ens.max(), 'xgb': xp, 'rf': rp}

def form_html(form):
    return '<div class="form-row">' + ''.join([
        f'<span class="fp fp-{r}">{r}</span>' for r in form
    ]) + '</div>'

def signal_badge_html(conf):
    if conf >= 0.65:
        return '<div class="signal-badge s-strong">⚡ STRONG SIGNAL</div>'
    elif conf >= 0.55:
        return '<div class="signal-badge s-mod">◈ MODERATE SIGNAL</div>'
    elif conf >= 0.45:
        return '<div class="signal-badge s-weak">⚠ WEAK SIGNAL</div>'
    else:
        return '<div class="signal-badge s-none">✕ NO CLEAR EDGE</div>'

def insider_notes(home, away, res, hs, as_, h2h):
    notes = []
    notes.append(("🤖", f"Ensemble model: XGBoost ({res['xgb'][np.argmax(res['proba'])]*100:.0f}%) + Random Forest ({res['rf'][np.argmax(res['proba'])]*100:.0f}%) weighted 60/40."))
    if res['confidence'] > 0.60:
        notes.append(("🔥", f"High model conviction — {res['outcome']} is the dominant call across both algorithms."))
    else:
        notes.append(("🌊", "Probabilities are spread. Treat this as a contested fixture — anything can happen."))
    gd_diff = hs['gd'] - as_['gd']
    if abs(gd_diff) > 10:
        better = home if gd_diff > 0 else away
        notes.append(("📈", f"{better} hold a significantly better goal difference this season, indicating stronger overall form."))
    if hs['win_rate'] > 60:
        notes.append(("🏠", f"{home} are winning {hs['win_rate']}% of their recent matches — elite consistency at home."))
    if as_['win_rate'] > 60:
        notes.append(("✈️", f"{away} carry a {as_['win_rate']}% win rate — dangerous regardless of venue."))
    if h2h:
        notes.append(("⚔️", f"Historical record ({home} perspective): {h2h.count('W')}W · {h2h.count('D')}D · {h2h.count('L')}L from last {len(h2h)} meetings."))
    if res['proba'][1] > 0.30:
        notes.append(("🤝", f"Draw probability elevated ({res['proba'][1]*100:.0f}%). Expect a tight, cagey tactical battle."))
    notes.append(("⚠️", "Injuries, suspensions, weather, and referee decisions are NOT factored in. Always apply context before acting."))
    return notes

if 'page'   not in st.session_state: st.session_state.page   = 'landing'
if 'result' not in st.session_state: st.session_state.result = None

st.markdown("""
<div class="orb-container">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
<div class="bg-grid"></div>
""", unsafe_allow_html=True)

xgb_model, rf_model, feature_cols, models_ok = load_models()
df, teams, data_ok = load_data()

if st.session_state.page == 'landing':

    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;">
      <div class="eyebrow">
        <span class="eyebrow-pulse"></span>
        AI-POWERED · PREMIER LEAGUE · SEASON 2024/25
      </div>
    </div>

    <div class="hero-wordmark">
      <span class="wm-kick">KICK</span><span class="wm-iq">IQ</span>
    </div>

    <div class="hero-tagline">Predict the game — before the whistle blows</div>

    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-strip" style="max-width:700px;margin-left:auto;margin-right:auto;margin-top:4rem;margin-bottom:3.5rem;">
      <div class="metric-block" style="text-align:center;">
        <div class="metric-val">3,800+</div>
        <div class="metric-lbl">Matches Trained</div>
      </div>
      <div class="metric-block" style="text-align:center;">
        <div class="metric-val">219</div>
        <div class="metric-lbl">Features Engineered</div>
      </div>
      <div class="metric-block" style="text-align:center;">
        <div class="metric-val">10Y</div>
        <div class="metric-lbl">Historical Data</div>
      </div>
      <div class="metric-block" style="text-align:center;">
        <div class="metric-val">2</div>
        <div class="metric-lbl">Stacked Models</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.button("⚡  LAUNCH PREDICTION ENGINE", key="cta_btn"):
            st.session_state.page = 'predict'
            st.rerun()

    st.markdown("""
    <div class="vs-section" style="margin-top:5rem;">
      <div class="vs-heading">WHY KICKIQ HITS DIFFERENT</div>
      <div class="compare-grid">
        <div class="compare-col ours">
          <div class="compare-col-label">⚡ KICKIQ</div>
          <div class="compare-row"><span class="c-check">✓</span> 219 engineered features per match</div>
          <div class="compare-row"><span class="c-check">✓</span> XGBoost + Random Forest ensemble</div>
          <div class="compare-row"><span class="c-check">✓</span> 10 years of training data</div>
          <div class="compare-row"><span class="c-check">✓</span> Home/Away split analysis</div>
          <div class="compare-row"><span class="c-check">✓</span> Head-to-head historical patterns</div>
          <div class="compare-row"><span class="c-check">✓</span> Momentum & form streaks tracked</div>
          <div class="compare-row"><span class="c-check">✓</span> Goal difference trend analysis</div>
          <div class="compare-row"><span class="c-check">✓</span> Confidence-scored signals</div>
          <div class="compare-row"><span class="c-check">✓</span> AI insider context per match</div>
        </div>
        <div class="compare-divider"><div class="vs-pill">VS</div></div>
        <div class="compare-col">
          <div class="compare-col-label" style="color:rgba(244,244,245,0.3);"> Other Predictors</div>
          <div class="compare-row"><span class="c-cross">✗</span> Single model only</div>
          <div class="compare-row"><span class="c-cross">✗</span> Only recent form</div>
          <div class="compare-row"><span class="c-cross">✗</span> No ensemble stacking</div>
          <div class="compare-row"><span class="c-cross">✗</span> Basic win/loss stats</div>
          <div class="compare-row"><span class="c-cross">✗</span> No H2H weighting</div>
          <div class="compare-row"><span class="c-cross">✗</span> Static output only</div>
          <div class="compare-row"><span class="c-cross">✗</span> No trend analysis</div>
          <div class="compare-row"><span class="c-cross">✗</span> No confidence scoring</div>
          <div class="compare-row"><span class="c-cross">✗</span> No contextual insights</div>
        </div>
      </div>
    </div>

    <div class="landing-footer">
      Built with XGBoost · scikit-learn · Streamlit · 2024/25 EPL Data<br>
      ⚠ For entertainment purposes only · Bet responsibly · Not financial advice
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  

elif st.session_state.page == 'predict':

    if not models_ok or not data_ok:
        st.error(" Run `python src/train_models.py` first to generate models & features.")
        st.stop()

    st.markdown("""
    <div class="pred-topbar">
      <div>
        <div class="pred-logo-text">KICK<span class="iq">IQ</span></div>
        <div class="pred-subtitle"><span class="live-dot"></span>PREDICTION ENGINE · LIVE</div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.14em;color:rgba(244,244,245,0.25);text-transform:uppercase;">
          EPL 2024/25
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    
    back_col1, back_col2 = st.columns([8,2])
    with back_col2:
        if st.button("← BACK", key="back_btn"):
            st.session_state.page   = 'landing'
            st.session_state.result = None
            st.rerun()

   
    st.markdown('<div class="selector-zone">', unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:950px;margin:auto;">
      <div class="section-kicker">MATCH SELECTION</div>
      <div class="section-title" style="text-align:center;">WHO'S PLAYING?</div>
    </div>
    """, unsafe_allow_html=True)

    outer_left, center_area, outer_right = st.columns([1,7,1])

    with center_area:

        c1, c2, c3 = st.columns([5,1,5])

        with c1:
            home_team = st.selectbox("🏠  HOME TEAM", teams, key="home_sel")

        with c2:
            st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center;">
                <span style="font-family:'Anton';font-size:1.9rem;color:rgba(244,244,245,0.15);">VS</span>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            away_options = [t for t in teams if t != home_team]
            away_team = st.selectbox("✈️  AWAY TEAM", away_options, key="away_sel")

    st.markdown('</div>', unsafe_allow_html=True)

    
    cta_l, cta_m, cta_r = st.columns([2,4,2])

    with cta_m:
        clicked = st.button(
            "⚡  ANALYSE THIS MATCH",
            key="pred_btn",
            use_container_width=True
        )

    if clicked:
        with st.spinner('Running ensemble analysis…'):
            st.session_state.result = run_prediction(
                home_team,
                away_team,
                xgb_model,
                rf_model,
                df,
                feature_cols
            )

    if st.session_state.result:
        
        res = st.session_state.result
        hs  = get_team_stats(df, home_team)
        as_ = get_team_stats(df, away_team)
        hf  = get_team_form(df, home_team, 5)
        af  = get_team_form(df, away_team, 5)
        h2h = get_h2h(df, home_team, away_team, 5)

        pad = '<div style="padding:0 3rem 5rem;position:relative;z-index:1;">'
        st.markdown(pad, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-hero">
          <div class="result-eyebrow">{home_team.upper()} vs {away_team.upper()} · {datetime.now().strftime('%d %B %Y').upper()}</div>
          <div class="result-outcome-text">{res['outcome'].upper()}</div>
          <div class="result-matchup-text">MODEL PREDICTION · ENSEMBLE</div>
          {signal_badge_html(res['confidence'])}
          <div class="disclaimer-txt">⚠ Entertainment only · Not financial advice · Bet at your own risk</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-tag" style="margin-top:2rem;">PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)

        ha = "active" if res['outcome']=='Home Win' else ""
        da = "active" if res['outcome']=='Draw'     else ""
        aa = "active" if res['outcome']=='Away Win' else ""

        st.markdown(f"""
        <div class="prob-grid">
          <div class="prob-card {ha}">
            <div class="prob-pct">{res['proba'][0]*100:.0f}%</div>
            <div class="prob-label">🏠 {home_team}</div>
          </div>
          <div class="prob-card {da}">
            <div class="prob-pct">{res['proba'][1]*100:.0f}%</div>
            <div class="prob-label">🤝 Draw</div>
          </div>
          <div class="prob-card {aa}">
            <div class="prob-pct">{res['proba'][2]*100:.0f}%</div>
            <div class="prob-label">✈️ {away_team}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        winner_idx = int(np.argmax(res['proba']))
        bar_colors = [
            'rgba(200,255,0,0.85)' if i==winner_idx else 'rgba(255,255,255,0.06)'
            for i in range(3)
        ]
        fig = go.Figure(go.Bar(
            x=[home_team, 'Draw', away_team],
            y=[p*100 for p in res['proba']],
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                cornerradius=6
            ),
            text=[f"{p*100:.1f}%" for p in res['proba']],
            textposition='outside',
            textfont=dict(family='DM Mono', size=11, color='rgba(244,244,245,0.5)')
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(family='DM Mono', size=10, color='rgba(244,244,245,0.35)')
            ),
            bargap=0.4
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="line-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">TEAM INTELLIGENCE</div>', unsafe_allow_html=True)

        def gd_color(v): return "#C8FF00" if v >= 0 else "#FF3A3A"
        def gd_str(v):   return f"+{v}" if v >= 0 else str(v)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker">🏠 {home_team.upper()} · RECENT FORM</div>
              <div class="intel-metrics">
                <div class="intel-metric">
                  <div class="intel-metric-val">{hs['wins']}</div>
                  <div class="intel-metric-lbl">Wins</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val">{hs['draws']}</div>
                  <div class="intel-metric-lbl">Draws</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val">{hs['losses']}</div>
                  <div class="intel-metric-lbl">Losses</div>
                </div>
              </div>
              <div class="intel-metrics">
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:#C8FF00;">{hs['goals_for']}</div>
                  <div class="intel-metric-lbl">Goals For</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:#FF3A3A;">{hs['goals_against']}</div>
                  <div class="intel-metric-lbl">Goals Against</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:{gd_color(hs['gd'])};">{gd_str(hs['gd'])}</div>
                  <div class="intel-metric-lbl">Goal Diff</div>
                </div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.5rem;">LAST 5</div>
              {form_html(hf)}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="intel-card">
              <div class="intel-card-kicker">✈️ {away_team.upper()} · RECENT FORM</div>
              <div class="intel-metrics">
                <div class="intel-metric">
                  <div class="intel-metric-val">{as_['wins']}</div>
                  <div class="intel-metric-lbl">Wins</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val">{as_['draws']}</div>
                  <div class="intel-metric-lbl">Draws</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val">{as_['losses']}</div>
                  <div class="intel-metric-lbl">Losses</div>
                </div>
              </div>
              <div class="intel-metrics">
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:#C8FF00;">{as_['goals_for']}</div>
                  <div class="intel-metric-lbl">Goals For</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:#FF3A3A;">{as_['goals_against']}</div>
                  <div class="intel-metric-lbl">Goals Against</div>
                </div>
                <div class="intel-metric">
                  <div class="intel-metric-val" style="color:{gd_color(as_['gd'])};">{gd_str(as_['gd'])}</div>
                  <div class="intel-metric-lbl">Goal Diff</div>
                </div>
              </div>
              <div class="intel-card-kicker" style="margin-top:1.2rem;margin-bottom:0.5rem;">LAST 5</div>
              {form_html(af)}
            </div>
            """, unsafe_allow_html=True)

        if h2h:
            st.markdown('<div class="section-tag" style="margin-top:1.5rem;">HEAD TO HEAD</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="h2h-card">
              <div class="intel-card-kicker">⚔️ LAST {len(h2h)} MEETINGS · {home_team.upper()} PERSPECTIVE</div>
              {form_html(h2h)}
              <div class="h2h-stat">
                <span>📊</span>
                <span>{home_team} won {h2h.count('W')}, drew {h2h.count('D')}, lost {h2h.count('L')} of last {len(h2h)} meetings against {away_team}.</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-tag" style="margin-top:0.5rem;">INSIDER INTEL</div>', unsafe_allow_html=True)
        notes = insider_notes(home_team, away_team, res, hs, as_, h2h)
        notes_html = ''.join([
            f'<div class="notes-item"><span class="note-icon">{ic}</span><span>{tx}</span></div>'
            for ic, tx in notes
        ])
        st.markdown(f"""
        <div class="notes-card">
          <div class="intel-card-kicker">🤖 AI ANALYSIS · {home_team.upper()} vs {away_team.upper()}</div>
          {notes_html}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬  MODEL INTERNALS — XGBoost vs Random Forest"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**XGBoost (60% weight)**")
                for label, p in zip([home_team, 'Draw', away_team], res['xgb']):
                    st.markdown(f"`{label}` → **{p*100:.1f}%**")
            with c2:
                st.markdown("**Random Forest (40% weight)**")
                for label, p in zip([home_team, 'Draw', away_team], res['rf']):
                    st.markdown(f"`{label}` → **{p*100:.1f}%**")

        st.markdown('</div>', unsafe_allow_html=True)  # results pad