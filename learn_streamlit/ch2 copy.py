import streamlit as st
import pandas as pd
from datetime import datetime

# What is widgets !
# => Elements that are the part of user-interaction: button, select box, input, check box, etc
# Page configuration
st.set_page_config(
    page_title="Professional Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .feature-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0066cc;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select a section:", 
                    ["Dashboard", "Analytics", "Services", "Contact"])
    st.divider()
    st.markdown("### Quick Settings")
    theme = st.selectbox("Choose theme:", ["Professional", "Modern", "Classic"])

# Header
st.markdown("""
    <div class="main-header">
        <h1>Professional Dashboard</h1>
        <p>Interactive Analytics & Services Platform</p>
    </div>
""", unsafe_allow_html=True)

# DASHBOARD PAGE
if page == "Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Users", "12,453", "↑ 12.5%"),
        ("Revenue", "$84,320", "↑ 8.2%"),
        ("Growth Rate", "23.5%", "↑ 5.1%"),
        ("Active Projects", "45", "→ 0%")
    ]
    
    for i, (col, (label, value, change)) in enumerate(zip([col1, col2, col3, col4], metrics)):
        with col:
            st.metric(label=label, value=value, delta=change)
    
    st.divider()
    
    # Interactive data visualization
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Monthly Performance")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        sales = [65, 78, 72, 88, 95, 102]
        df = pd.DataFrame({'Month': months, 'Sales': sales})
        st.area_chart(df.set_index('Month'))
    
    with col_chart2:
        st.subheader("Revenue Distribution")
        categories = ['Product A', 'Product B', 'Product C', 'Product D']
        revenue = [35, 25, 20, 20]
        df_pie = pd.DataFrame({'Category': categories, 'Revenue': revenue})
        st.bar_chart(df_pie.set_index('Category'))
    
    st.divider()
    
    # Interactive table
    st.subheader("Recent Transactions")
    transactions_data = {
        'Date': ['2025-12-14', '2025-12-13', '2025-12-12', '2025-12-11'],
        'Customer': ['Acme Corp', 'Tech Solutions', 'Global Industries', 'Digital Systems'],
        'Amount': ['$15,000', '$8,500', '$12,300', '$6,800'],
        'Status': ['Completed', 'Completed', 'Pending', 'Completed']
    }
    df_transactions = pd.DataFrame(transactions_data)
    st.dataframe(df_transactions, use_container_width=True)

# ANALYTICS PAGE
elif page == "Analytics":
    st.header("📈 Advanced Analytics")
    
    # Interactive controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    
    with col_ctrl1:
        metric_select = st.selectbox("Select Metric:", ["Revenue", "Users", "Engagement", "Conversion"])
    with col_ctrl2:
        time_range = st.selectbox("Time Range:", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year-to-Date"])
    
    # Sample data
    import numpy as np
    dates = pd.date_range(start='2025-11-15', periods=30)
    analytics_data = pd.DataFrame({
        'Date': dates,
        'Revenue': np.random.randint(5000, 15000, 30),
        'Users': np.random.randint(100, 500, 30),
        'Engagement': np.random.randint(30, 90, 30)
    })
    
    st.line_chart(analytics_data.set_index('Date'))
    
    st.divider()
    
    # Detailed metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="metric-card"><h4>Avg. Daily Revenue</h4><h2>$10,234</h2></div>', 
                   unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card"><h4>Avg. Daily Users</h4><h2>342</h2></div>', 
                   unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card"><h4>Engagement Rate</h4><h2>62.3%</h2></div>', 
                   unsafe_allow_html=True)

# SERVICES PAGE
elif page == "Services":
    st.header("🎯 Our Services")
    
    services = [
        {
            "name": "Data Analytics",
            "desc": "Advanced analytics and reporting solutions",
            "price": "$999/mo"
        },
        {
            "name": "Cloud Integration",
            "desc": "Seamless cloud infrastructure setup",
            "price": "$1,499/mo"
        },
        {
            "name": "24/7 Support",
            "desc": "Dedicated support team",
            "price": "$499/mo"
        },
        {
            "name": "Custom Development",
            "desc": "Tailored solutions for your business",
            "price": "Custom"
        }
    ]
    
    cols = st.columns(2)
    for i, service in enumerate(services):
        with cols[i % 2]:
            st.markdown(f"""
                <div class="feature-box">
                    <h4>{service['name']}</h4>
                    <p>{service['desc']}</p>
                    <h3 style="color: #667eea;">{service['price']}</h3>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Learn More: {service['name']}", key=f"btn_{i}"):
                st.info(f"Details about {service['name']} - Coming soon!")
    
    st.divider()
    
    # Interactive service selector
    st.subheader("Interactive Service Configurator")
    selected_services = st.multiselect(
        "Select services you're interested in:",
        [s['name'] for s in services]
    )
    
    if selected_services:
        st.success(f"You've selected: {', '.join(selected_services)}")
        with st.expander("View Package Details"):
            for svc in selected_services:
                st.write(f"✓ {svc}")

# CONTACT PAGE
else:
    st.header("📧 Get In Touch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contact Information")
        st.write("📍 123 Business Street, Suite 100")
        st.write("📞 +1 (555) 123-4567")
        st.write("📧 hello@company.com")
        st.write("🕐 Monday - Friday: 9AM - 6PM EST")
        
        st.divider()
        st.subheader("Connect With Us")
        social_cols = st.columns(4)
        socials = ["LinkedIn", "Twitter", "Facebook", "Instagram"]
        for col, social in zip(social_cols, socials):
            with col:
                if st.button(f"🔗 {social}", use_container_width=True):
                    st.info(f"Opening {social}...")
    
    with col2:
        st.subheader("Send us a Message")
        
        with st.form("contact_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email Address", placeholder="john@example.com")
            phone = st.text_input("Phone (Optional)", placeholder="+1 (555) 123-4567")
            subject = st.selectbox("Subject", 
                                  ["General Inquiry", "Support", "Sales", "Partnership", "Other"])
            message = st.text_area("Message", placeholder="Tell us how we can help...", height=150)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("Send Message", use_container_width=True)
            with col_btn2:
                st.form_submit_button("Clear", use_container_width=True)
            
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you! Your message has been sent successfully.")
                    st.balloons()
                    st.write(f"We'll contact you at {email} shortly.")
                else:
                    st.error("❌ Please fill in all required fields (Name, Email, Message)")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>© 2025 Professional Dashboard. All rights reserved.</p>
        <p>Last updated: """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """</p>
    </div>
""", unsafe_allow_html=True)