import streamlit as st
import requests

st.set_page_config(
    page_title="HDB Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 HDB Resale Price Prediction")
st.markdown("---")

tab1, tab2= st.tabs(["📈 Prediction", "🤖 Chatbot"])

with tab1:
    st.subheader("Enter Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        town = st.text_input("🏘️ Town", placeholder="e.g., ANG MO KIO")

    with col2:
        flat_type = st.selectbox("🏠 Flat Type", 
            ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])

    with col3:
        storey = st.selectbox("🏢 Storey Level", ["low", "medium", "high"])

    st.info("**Storey Guide:** Low (1-5 floors) | Medium (6-20 floors) | High (21+ floors)")

    predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        if not town.strip():
            st.error("❌ Please enter a town name")
        else:
            with st.spinner("Filtering data, training model, and making prediction..."):
                try:
                    # Single API call that does everything
                    prediction_request = {
                        "town": town.strip().upper(),
                        "flat_type": flat_type,
                        "storey_range_classify": storey,
                        "lease_remaining": 70.0,
                        "floor_area_sqm": 75.0
                    }
                    
                    response = requests.post("http://127.0.0.1:8000/predict", 
                        json=prediction_request, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            price = result["predicted_price"]
                            training_records = result["training_records"]
                            
                            st.success("🎯 Prediction Complete!")
                            st.metric("💰 Predicted Price", f"${price:,.2f}")
                            st.info(f"📍 {town.upper()} | {flat_type} | {storey.title()} Floor | Based on {training_records} records")
                    else:
                        st.error("❌ Prediction request failed")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to server. Make sure API is running on port 8000.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab2:
    st.subheader("🤖 HDB AI Assistant")
    st.markdown("Ask me anything about HDB housing, prices, policies, or recommendations!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about HDB housing..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/chat",
                        json={"message": prompt}, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "error" in result:
                            bot_response = f"Sorry, I encountered an error: {result['error']}"
                        else:
                            bot_response = result["response"]
                    else:
                        bot_response = "Sorry, I'm having trouble connecting. Please try again."
                        
                except Exception as e:
                    bot_response = f"Connection error: {str(e)}"
                
                st.markdown(bot_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Sample questions
    st.markdown("---")

        
    if st.button("🗑️ Clear Chat", key="clear"):
            st.session_state.messages = []
            st.rerun()

st.markdown("---")
st.caption("HDB Price Prediction | Enter Town, Flat Type & Storey Level")