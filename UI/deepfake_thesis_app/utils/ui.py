import streamlit as st


def render_professional_sidebar(page_title: str) -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                line-height: 1.3;
            }
            .thesis-sidebar-card {
                border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
                border-radius: 12px;
                padding: 0.85rem;
                background: color-mix(in srgb, var(--secondary-background-color) 85%, white 15%);
                margin-bottom: 0.8rem;
            }
            .thesis-sidebar-title {
                font-weight: 600;
                margin-bottom: 0.15rem;
            }
            .thesis-sidebar-caption {
                font-size: 0.85rem;
                opacity: 0.85;
            }
            .thesis-section-card {
                border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
                border-radius: 12px;
                padding: 0.7rem;
                margin-bottom: 0.7rem;
                background: color-mix(in srgb, var(--secondary-background-color) 90%, white 10%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _ = page_title
