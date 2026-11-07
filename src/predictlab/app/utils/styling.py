"""CSS styles and visual design helpers for PredictLab analytical interface."""

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom SaaS styling for dark analytical dashboard theme."""
    st.markdown(
        """
        <style>
        /* Modern analytical typography and card styling */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }
        .metric-card-label {
            font-size: 0.85rem;
            color: #94A3B8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-card-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .metric-card-delta {
            font-size: 0.85rem;
            margin-top: 6px;
            font-weight: 500;
        }
        .delta-positive {
            color: #34D399;
        }
        .delta-neutral {
            color: #94A3B8;
        }
        .delta-warning {
            color: #F87171;
        }

        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .risk-low {
            background-color: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .risk-medium {
            background-color: rgba(251, 191, 36, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(251, 191, 36, 0.35);
        }
        .risk-high {
            background-color: rgba(244, 63, 94, 0.15);
            color: #F43F5E;
            border: 1px solid rgba(244, 63, 94, 0.35);
        }

        /* Status Pills */
        .status-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.80rem;
            font-weight: 600;
        }
        .status-retained {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-churn {
            background-color: rgba(249, 115, 22, 0.15);
            color: #F97316;
            border: 1px solid rgba(249, 115, 22, 0.3);
        }

        /* Clean Information Card */
        .info-card {
            background-color: #1E293B;
            border-left: 4px solid #2563EB;
            border-radius: 0 8px 8px 0;
            padding: 16px 20px;
            margin: 14px 0;
            color: #E2E8F0;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .warning-card {
            background-color: rgba(245, 158, 11, 0.08);
            border-left: 4px solid #F59E0B;
            border-radius: 0 8px 8px 0;
            padding: 14px 18px;
            margin: 12px 0;
            color: #FDE68A;
            font-size: 0.90rem;
            line-height: 1.45;
        }

        /* Section Header Divider */
        .section-header {
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
