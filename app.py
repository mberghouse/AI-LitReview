import streamlit as st
from literature_agent import LitReviewPapersAgent
from citation_alignment_agent import CitationAlignmentAgent

def show_sidebar_references(papers, summaries):
    """
    Renders a sidebar with references, each of which includes:
      - Paper title (linked to an external URL, if available)
      - A short snippet or highlight from the paper
    """
    st.sidebar.title("References Used")
    summary_dict = {}
    
    # Parse summaries into a dictionary
    for line in summaries.split('\n'):
        if line.strip():
            num = line[1:line.index(']')]
            text = line[line.index(']')+1:].strip()
            summary_dict[num] = text
    
    for i, paper in enumerate(papers, start=1):
        with st.sidebar.expander(f"[{i}] {paper['title']}"):
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Year:** {paper['date']}")
            
            # Handle different URL formats
            url = None
            if paper.get('doi'):
                url = f"https://doi.org/{paper['doi']}"
            elif paper.get('pubmed_id'):
                url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pubmed_id']}"
            elif paper.get('url'):
                url = paper['url']
                
            if url:
                st.markdown(f"[Link to Paper]({url})")
            
            st.write("---")
            st.write(f"**Key Points:** {summary_dict.get(str(i), '...')}")

def create_bibliography(papers):
    """Create a properly formatted bibliography from ordered papers."""
    bibliography = []
    for i, paper in enumerate(papers, start=1):
        authors = paper['authors']
        title = paper['title']
        date = paper['date']
        url = None
        
        if paper.get('doi'):
            url = f"https://doi.org/{paper['doi']}"
        elif paper.get('pubmed_id'):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pubmed_id']}"
        elif paper.get('url'):
            url = paper['url']
        
        ref = f"[{i}] {authors}. ({date}). {title}."
        if url:
            ref += f" Available at: {url}"
        
        bibliography.append(ref + "\n")  # Add newline after each reference
    
    return "\n".join(bibliography)

async def main():
    st.title("Literature Review Generator")
    
    # Add very visible starting message
    print("\n" + "="*50)
    print("THIS IS THE START!!!!!! LOOK HERE!!!!!!!!!!")
    print("="*50 + "\n")
    
    st.write("""
    This app generates a comprehensive literature review on your chosen topic using AI.
    Enter your topic of interest and specify how many references you'd like included.
    """)

    # Initialize session state for min_references if not exists
    if 'min_references' not in st.session_state:
        st.session_state.min_references = 5

    # Input fields
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    search_method = st.radio(
        "Choose search method:",
        ["PubMed Search", "Local Paper Database"],
        help="PubMed Search uses live PubMed data. Local Paper Database uses pre-downloaded papers."
    )
    
    review_type = st.radio(
        "Choose review type:",
        ["Quick Review", "Standard Review", "Deep Review"],
        help="""
        Quick Review: Faster but less detailed (GPT-4o-mini)
        Standard Review: Balanced speed and detail (GPT-4o)
        Deep Review: Most detailed but slower (o3-mini)
        """
    )
    
    # Set model based on review type
    if review_type == "Quick Review":
        model = "gpt-4o-mini"
        model_params = {"temperature": 0, "max_tokens": 16384}
    elif review_type == "Standard Review":
        model = "gpt-4o"
        model_params = {"temperature": 0, "max_tokens": 16384}
    else:  # Deep Review
        model = "o3-mini-2025-01-31"
        model_params = {}  # o3 models don't use temperature or max_tokens
    
    topic = st.text_area(
        label="Enter your research topic:",
        placeholder="Example: The impact of artificial intelligence on healthcare diagnostics",
        height=100
    )
    
    # Replace the slider section with this:
    min_refs = st.slider(
        "Minimum number of references to include:",
        min_value=3,
        max_value=100,
        value=st.session_state.min_references,
        help="Select the minimum number of references you want in your literature review",
        key="temp_min_refs"  # Use a temporary key
    )

    if st.button("Generate Literature Review"):
        if not openai_api_key:
            st.warning("Please enter a valid OpenAI API Key to proceed.")
            return
        
        if not topic:
            st.warning("Please enter a research topic.")
            return
        
        # Update session state only when generating
        st.session_state.min_references = min_refs
        
        # Use the stored min_references value
        agent = LitReviewPapersAgent(
            openai_api_key=openai_api_key, 
            openai_model=model,
            model_params=model_params,
            min_references=st.session_state.min_references,
            search_method=search_method
        )

        with st.spinner("Generating literature review..."):
            review_placeholder = st.empty()
            review_placeholder.markdown("Extracting keywords and fetching papers...")
            
            # Await the async run method
            review_text = await agent.run(topic)
            
            # Show the complete review
            review_placeholder.markdown(review_text)
            st.success("Literature review generated!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
