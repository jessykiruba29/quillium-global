import streamlit as st
import fitz  # PyMuPDF
import random, re, os, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Download WordNet for synonyms
nltk.download('wordnet')

st.set_page_config(page_title="Quillium", page_icon="🌿", layout="wide")

# ---------------- Cached Translator ----------------
@st.cache_resource(show_spinner=False)
def load_translator():
    try:
        model_name = "Helsinki-NLP/opus-mt-en-mul"
        local_dir = "./models/opus-mt-en-mul"
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(os.path.join(local_dir, "config.json")):
            tok = AutoTokenizer.from_pretrained(local_dir)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
        else:
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tok.save_pretrained(local_dir)
            mdl.save_pretrained(local_dir)
        return tok, mdl
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

# Initialize translator only once
if 'translator' not in st.session_state:
    with st.spinner("🔄 Loading translation model..."):
        translator_tokenizer, translator_model = load_translator()
        st.session_state.translator_tokenizer = translator_tokenizer
        st.session_state.translator_model = translator_model

def translate_text(text, target_lang):
    if target_lang == "English" or not text.strip():
        return text
    
    if st.session_state.translator_tokenizer is None:
        return text + " [Translation unavailable]"
    
    # Expanded language prefix mapping for global languages
    prefix_map = {
        # European languages
        "Spanish": ">>spa<< ", "French": ">>fra<< ", "German": ">>deu<< ",
        "Italian": ">>ita<< ", "Portuguese": ">>por<< ", "Russian": ">>rus<< ",
        "Dutch": ">>nld<< ", "Polish": ">>pol<< ", "Ukrainian": ">>ukr<< ",
        "Romanian": ">>ron<< ", "Greek": ">>ell<< ", "Czech": ">>ces<< ",
        "Swedish": ">>swe<< ", "Norwegian": ">>nor<< ", "Danish": ">>dan<< ",
        "Finnish": ">>fin<< ", "Hungarian": ">>hun<< ", "Bulgarian": ">>bul<< ",
        
        # Asian languages
        "Chinese": ">>zho<< ", "Japanese": ">>jpn<< ", "Korean": ">>kor<< ",
        "Arabic": ">>ara<< ", "Hebrew": ">>heb<< ", "Turkish": ">>tur<< ",
        "Thai": ">>tha<< ", "Vietnamese": ">>vie<< ", "Indonesian": ">>ind<< ",
        "Malay": ">>msa<< ", "Filipino": ">>tgl<< ", "Persian": ">>fas<< ",
        
        # Indian languages (kept for completeness)
        "Hindi": ">>hin<< ", "Tamil": ">>tam<< ", "Telugu": ">>tel<< ",
        "Kannada": ">>kan<< ", "Malayalam": ">>mal<< ", "Bengali": ">>ben<< ",
        "Marathi": ">>mar<< ", "Gujarati": ">>guj<< ", "Punjabi": ">>pan<< ",
        "Urdu": ">>urd<< ",
        
        # Additional languages
        "Swahili": ">>swa<< ", "Zulu": ">>zul<< ", "Afrikaans": ">>afr<< ",
        "Catalan": ">>cat<< ", "Croatian": ">>hrv<< ", "Serbian": ">>srp<< ",
        "Slovak": ">>slk<< ", "Slovenian": ">>slv<< ", "Lithuanian": ">>lit<< ",
        "Latvian": ">>lav<< ", "Estonian": ">>est<< ", "Maltese": ">>mlt<< ",
        "Icelandic": ">>isl<< "
    }
    
    prefix = prefix_map.get(target_lang, "")
    
    try:
        # More efficient translation - shorter text for better performance
        if len(text) > 150:
            text = text[:150] + "..."
            
        inputs = st.session_state.translator_tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512)
        outputs = st.session_state.translator_model.generate(**inputs, max_new_tokens=100)
        return st.session_state.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"{text} [Translation failed]"

# ---------------- Cached PDF Processing ----------------
@st.cache_data(show_spinner="📄 Processing PDF...", ttl=3600)
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text("text").strip()
            if page_text:
                full_text += page_text + " "
        
        page_count = doc.page_count
        doc.close()
        
        if len(full_text.strip()) < 50:
            return "This document contains minimal text. Please try a document with more content.", page_count
            
        return full_text.strip(), page_count
    except Exception as e:
        return f"Error processing PDF: {e}", 0

@st.cache_data(show_spinner=False, ttl=1800)
def extract_questions_from_text(text):
    if text.startswith("Error") or text.startswith("This document contains"):
        return []
        
    questions = []
    # More robust sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Process more sentences for 20 questions
    sentences = sentences[:200]  # Increased to 200 sentences

    question_patterns = [
        # Definition patterns
        (r"^(.*?)\s+is\s+(?:an?\s+)?(.*)", "What is {}?"),
        (r"^(.*?)\s+are\s+(.*)", "What are {}?"),
        (r"^(.*?)\s+means\s+(.*)", "What does {} mean?"),
        (r"^(.*?)\s+refers to\s+(.*)", "What does {} refer to?"),
        (r"^(.*?)\s+can be defined as\s+(.*)", "How is {} defined?"),
        (r"^(.*?)\s+is defined as\s+(.*)", "How is {} defined?"),
        (r"^(.*?)\s+known as\s+(.*)", "What is {} known as?"),
        (r"^(.*?)\s+called\s+(.*)", "What is {} called?"),
        
        # Process/Description patterns
        (r"^(.*?)\s+involves\s+(.*)", "What does {} involve?"),
        (r"^(.*?)\s+includes\s+(.*)", "What does {} include?"),
        (r"^(.*?)\s+consists of\s+(.*)", "What does {} consist of?"),
        (r"^(.*?)\s+occurs when\s+(.*)", "When does {} occur?"),
        (r"^(.*?)\s+happens when\s+(.*)", "When does {} happen?"),
        
        # Purpose/Function patterns
        (r"^(.*?)\s+is used to\s+(.*)", "What is {} used for?"),
        (r"^(.*?)\s+is used for\s+(.*)", "What is {} used for?"),
        (r"^(.*?)\s+helps to\s+(.*)", "How does {} help?"),
        (r"^(.*?)\s+allows\s+(.*)", "What does {} allow?"),
        
        # Characteristic patterns
        (r"^(.*?)\s+has\s+(.*)", "What does {} have?"),
        (r"^(.*?)\s+contains\s+(.*)", "What does {} contain?"),
        (r"^(.*?)\s+provides\s+(.*)", "What does {} provide?"),
    ]

    used_sentences = set()
    
    for sent in sentences:
        if len(questions) >= 20:  # Stop when we have 20 questions
            break
            
        sent = sent.strip()
        if len(sent.split()) < 5 or len(sent) < 20:  # Slightly longer sentences
            continue
            
        if sent in used_sentences:
            continue
            
        # Try all patterns
        for pattern, question_template in question_patterns:
            match = re.match(pattern, sent, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                answer = match.group(2).strip()
                
                # Clean up subject and answer
                subject = re.sub(r'[.,;:]$', '', subject)
                answer = re.sub(r'[.,;:]$', '', answer)
                
                # Quality checks
                if (len(subject.split()) >= 1 and len(subject.split()) <= 8 and
                    len(answer.split()) >= 2 and len(answer) > 10):
                    
                    question_text = question_template.format(subject)
                    questions.append({
                        "question": question_text,
                        "answer": answer,
                        "source_sentence": sent
                    })
                    used_sentences.add(sent)
                    break

    # Enhanced fallback for more questions
    if len(questions) < 20:
        remaining_slots = 20 - len(questions)
        important_sentences = [
            s for s in sentences 
            if s not in used_sentences 
            and len(s.split()) > 8 
            and len(s) > 30
        ][:remaining_slots * 2]  # Get more candidates
        
        question_types = [
            "What is the main idea?",
            "What key concept is described?",
            "What important point is made?",
            "What is being explained?",
            "What information is provided?",
            "What does this describe?",
            "What process is outlined?",
            "What principle is discussed?"
        ]
        
        for i, sent in enumerate(important_sentences[:remaining_slots]):
            if len(questions) >= 20:
                break
            q_type = question_types[i % len(question_types)]
            questions.append({
                "question": q_type,
                "answer": sent,
                "source_sentence": sent
            })
    
    return questions[:20]  # Ensure exactly 20 questions

# ---------------- Efficient Smart distractors ----------------
def generate_smart_distractors(answer, n=3):
    if not answer or len(answer.split()) == 0:
        return ["Incorrect option A", "Wrong choice B", "Alternative C"]
        
    words = answer.split()
    distractors = set()

    # More efficient synonym replacement
    attempts = min(20, len(words) * 2)
    for _ in range(attempts):
        if len(distractors) >= n:
            break
            
        new_words = words.copy()
        idx = random.randint(0, len(words)-1)
        word = words[idx]
        
        # Skip very short words
        if len(word) <= 2:
            continue
            
        try:
            syns = wordnet.synsets(word)
            if syns:
                # Get multiple synonyms for variety
                lemmas = []
                for syn in syns[:2]:  # Limit to first 2 synsets
                    for lemma in syn.lemmas()[:2]:  # First 2 lemmas per synset
                        lemma_name = lemma.name().replace("_", " ")
                        if (lemma_name.lower() != word.lower() and 
                            len(lemma_name.split()) == 1 and
                            len(lemma_name) > 2):
                            lemmas.append(lemma_name)
                
                if lemmas:
                    new_words[idx] = random.choice(lemmas)
                    fake = " ".join(new_words)
                    if 10 < len(fake) < 100:  # Reasonable length
                        distractors.add(fake)
        except:
            pass

    # Better fallback distractors
    filler_groups = [
        ["A completely different approach", "Alternative methodology", "Various other techniques"],
        ["Multiple systems exist", "Several processes are possible", "Different methods apply"],
        ["Other frameworks exist", "Various alternatives available", "Different concepts apply"],
        ["Not the correct method", "Incorrect approach", "Wrong technique"],
        ["Opposite concept applies", "Contrary principle", "Different perspective"]
    ]
    
    while len(distractors) < n:
        group = random.choice(filler_groups)
        distractors.update(group)
    
    return list(distractors)[:n]

@st.cache_data(show_spinner="🔄 Loading...", ttl=1800)
def make_mcqs(text, lang="English", max_questions=20):  # Increased to 20
    qa_pairs = extract_questions_from_text(text)[:max_questions]
    if not qa_pairs:
        return []
        
    mcqs = []
    progress_bar = st.progress(0)
    
    for i, qa in enumerate(qa_pairs):
        # Update progress
        progress_bar.progress((i + 1) / len(qa_pairs))
        
        # Efficient translation - only translate what's necessary
        q = translate_text(qa["question"], lang)
        a = translate_text(qa["answer"], lang)
        
        # Generate distractors from original English for consistency
        original_distractors = generate_smart_distractors(qa["answer"])
        distractors = [translate_text(d, lang) for d in original_distractors]
        
        options = distractors + [a]
        random.shuffle(options)
        mcqs.append({"question": q, "answer": a, "options": options})
    
    progress_bar.empty()
    return mcqs

@st.cache_data(show_spinner="🔄 Creating flashcards...", ttl=1800)
def make_flashcards(text, lang="English", max_cards=20):  # Increased to 20
    qa_pairs = extract_questions_from_text(text)[:max_cards]
    flashcards = []
    
    for qa in qa_pairs:
        question = translate_text(qa["question"], lang)
        answer = translate_text(qa["answer"], lang)
        flashcards.append({"question": question, "answer": answer})
    
    return flashcards

# ---------------- Progress Tracking ----------------
def init_progress():
    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = {
            'total_questions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'quizzes_taken': 0,
            'flashcards_studied': 0
        }

def update_progress(correct=False, flashcard_studied=False):
    init_progress()
    if correct is not None:
        st.session_state.quiz_progress['total_questions'] += 1
        if correct:
            st.session_state.quiz_progress['correct_answers'] += 1
        else:
            st.session_state.quiz_progress['incorrect_answers'] += 1
    if flashcard_studied:
        st.session_state.quiz_progress['flashcards_studied'] += 1

# ---------------- Optimized Main App ----------------
def main():
    st.markdown("""
    <h1 style='color: #38ef7d; text-align: center; font-size: 3em; margin-bottom: 0;'>🌿Quillium</h1>
    <p style='color: #666; text-align: center; font-size: 1.2em; margin-top: 0;'>Quiz.Learn.Conquer</p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_progress()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Quiz"
    if 'flashcard_index' not in st.session_state:
        st.session_state.flashcard_index = 0
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False
    if 'processed_file_id' not in st.session_state:
        st.session_state.processed_file_id = None

    # Sidebar navigation
    st.sidebar.markdown("""
    <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #38ef7d; color: white;'>
        <h2 style='color: #38ef7d; margin-bottom: 20px;'>🌍 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Go to", ["Quiz", "Flashcards", "Progress"], label_visibility="collapsed")
    st.session_state.current_page = page

    # Global languages organized by region
    global_languages = {
        "English": "English",
        "European Languages": [
            "Spanish", "French", "German", "Italian", "Portuguese", 
            "Russian", "Dutch", "Polish", "Ukrainian", "Romanian",
            "Greek", "Czech", "Swedish", "Norwegian", "Danish",
            "Finnish", "Hungarian", "Bulgarian"
        ],
        "Asian Languages": [
            "Chinese", "Japanese", "Korean", "Arabic", "Hebrew",
            "Turkish", "Thai", "Vietnamese", "Indonesian", "Malay",
            "Filipino", "Persian", "Hindi", "Tamil", "Telugu",
            "Kannada", "Malayalam", "Bengali", "Marathi", "Gujarati",
            "Punjabi", "Urdu"
        ],
        "Other Languages": [
            "Swahili", "Zulu", "Afrikaans", "Catalan", "Croatian",
            "Serbian", "Slovak", "Slovenian", "Lithuanian", "Latvian",
            "Estonian", "Maltese", "Icelandic"
        ]
    }
    
    # Create language selection with groups
    lang_choice = st.sidebar.selectbox(
        "🌐 Choose Language",
        options=[global_languages["English"]] + 
                global_languages["European Languages"] + 
                global_languages["Asian Languages"] + 
                global_languages["Other Languages"],
        index=0
    )
    
    # Add question count slider
    question_count = st.sidebar.slider("Number of Questions", min_value=5, max_value=20, value=20)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #38ef7d; color: #ccc;'>
        <h4 style='color: #38ef7d;'>📚 How to use:</h4>
        <p>1. Upload a PDF document<br>
        2. Choose your language<br>
        3. Select number of questions<br>
        4. Start learning with quizzes or flashcards!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

    # CSS remains the same...
    st.markdown("""
    <style>
    .stApp { background: #0f1116; color: #e0e0e0; }
    h1, h2, h3 { color: #38ef7d !important; font-weight: 600; }
    .stButton>button {
        background: rgba(56, 239, 125, 0.1); color: #38ef7d; border: 1px solid #38ef7d; border-radius: 8px; padding: 8px 16px; font-weight: 500; transition: all 0.3s ease;
    }
    .stButton>button:hover { background: #38ef7d; color: #0f1116; border-color: #38ef7d; transform: translateY(-1px); }
    .flashcard-container {
        background: linear-gradient(135deg, rgba(56, 239, 125, 0.05) 0%, rgba(0, 0, 0, 0) 100%); border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 16px; padding: 40px; margin: 20px auto; min-height: 250px; display: flex; align-items: center; justify-content: center; text-align: center; transition: all 0.3s ease;
    }
    .progress-card { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin: 10px; text-align: center; transition: all 0.3s ease; }
    .progress-card h2 { color: #38ef7d !important; font-size: 2em; margin: 10px 0; }
    .progress-card h3 { color: #ccc !important; font-size: 0.9em; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

    if uploaded_file:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.processed_file_id != current_file_id:
            st.session_state.processed_file_id = current_file_id
            st.cache_data.clear()
        
        with st.spinner("📄 Processing your document..."):
            raw_text, page_count = extract_text_from_pdf(uploaded_file)
        
        # Show PDF info
        st.sidebar.markdown(f"""
        <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #38ef7d; color: #ccc;'>
            <h4 style='color: #38ef7d; margin-bottom: 15px;'>📊 Document Info</h4>
            <p style='margin: 5px 0;'>📄 <strong>Pages:</strong> {page_count}</p>
            <p style='margin: 5px 0;'>📝 <strong>Characters:</strong> {len(raw_text)}</p>
            <p style='margin: 5px 0;'>🎯 <strong>Questions:</strong> {question_count}</p>
            <p style='margin: 5px 0;'>🌐 <strong>Language:</strong> {lang_choice}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(raw_text) > 50 and not raw_text.startswith("Error"):
            # Quiz Page
            if st.session_state.current_page == "Quiz":
                st.header("🎯 Quiz Mode")
                
                with st.spinner(f"🔄 Generating {question_count} questions in {lang_choice}..."):
                    mcqs = make_mcqs(raw_text, lang=lang_choice, max_questions=question_count)
                
                if mcqs:
                    st.success(f"✅ Generated {len(mcqs)} questions in {lang_choice}")

                    for i, q in enumerate(mcqs):
                        st.markdown(f"### ❓ Q{i+1}: {q['question']}")

                        key_answered = f"answered_{i}"
                        if key_answered not in st.session_state:
                            st.session_state[key_answered] = None

                        selected_option = st.radio(
                            f"Select an answer for Q{i+1}:",
                            q['options'],
                            key=f"radio_{i}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        
                        if selected_option:
                            st.session_state[key_answered] = selected_option
                            if selected_option == q['answer']:
                                st.success("🎉 Correct! Well done!")
                                update_progress(correct=True)
                            else:
                                st.error(f"❌ Incorrect! The correct answer is: **{q['answer']}**")
                                update_progress(correct=False)
                        
                        st.markdown("---")
                else:
                    st.warning("⚠️ No questions could be generated from this PDF. Try a document with more educational content.")

            # Flashcards Page
            elif st.session_state.current_page == "Flashcards":
                st.header("📚 Flashcard Mode")
                
                flashcards = make_flashcards(raw_text, lang=lang_choice, max_cards=question_count)
                
                if flashcards:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        current_card = flashcards[st.session_state.flashcard_index]
                        
                        card_content = current_card['answer'] if st.session_state.show_answer else current_card['question']
                        
                        st.markdown(
                            f'<div class="flashcard-container">'
                            f'<div style="font-size: 1.3em; color: #e0e0e0; font-weight: 500;">{card_content}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        if st.button("🔄 Flip Card", key="flip_btn", use_container_width=True):
                            st.session_state.show_answer = not st.session_state.show_answer
                            update_progress(flashcard_studied=True)
                        
                        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
                        
                        with nav_col1:
                            if st.button("⬅️ Previous", use_container_width=True):
                                st.session_state.flashcard_index = (st.session_state.flashcard_index - 1) % len(flashcards)
                                st.session_state.show_answer = False
                        
                        with nav_col2:
                            st.markdown(f"<p style='text-align: center; color: #ccc;'>📖 Card {st.session_state.flashcard_index + 1} of {len(flashcards)}</p>", unsafe_allow_html=True)
                        
                        with nav_col3:
                            if st.button("Next ➡️", use_container_width=True):
                                st.session_state.flashcard_index = (st.session_state.flashcard_index + 1) % len(flashcards)
                                st.session_state.show_answer = False
                    
                    st.info("💡 Use the Flip Card button to reveal the answer!")
                
                else:
                    st.warning("⚠️ No flashcards could be generated from this PDF.")

            # Progress Page (same as before)
            elif st.session_state.current_page == "Progress":
                st.header("📊 Your Learning Progress")
                progress = st.session_state.quiz_progress
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="progress-card">
                        <h3>Total Questions</h3>
                        <h2>{progress['total_questions']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    accuracy = (progress['correct_answers'] / progress['total_questions'] * 100) if progress['total_questions'] > 0 else 0
                    st.markdown(f"""
                    <div class="progress-card">
                        <h3>Accuracy</h3>
                        <h2>{accuracy:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="progress-card">
                        <h3>Correct Answers</h3>
                        <h2>{progress['correct_answers']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="progress-card">
                        <h3>Flashcards Studied</h3>
                        <h2>{progress['flashcards_studied']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                if progress['total_questions'] > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        accuracy = (progress['correct_answers'] / progress['total_questions'] * 100)
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = accuracy,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Accuracy Score", 'font': {'color': '#38ef7d', 'size': 20}},
                            delta = {'reference': 50, 'increasing': {'color': "#38ef7d"}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#38ef7d"},
                                'bar': {'color': "#38ef7d"},
                                'bgcolor': "rgba(255, 255, 255, 0.05)",
                                'borderwidth': 2,
                                'bordercolor': "#38ef7d",
                                'steps': [
                                    {'range': [0, 50], 'color': 'rgba(255, 107, 107, 0.3)'},
                                    {'range': [50, 80], 'color': 'rgba(255, 193, 7, 0.3)'},
                                    {'range': [80, 100], 'color': 'rgba(56, 239, 125, 0.3)'}],
                                'threshold': {
                                    'line': {'color': "#38ef7d", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90}}
                        ))
                        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#e0e0e0"})
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        if progress['correct_answers'] > 0 or progress['incorrect_answers'] > 0:
                            fig_pie = px.pie(
                                values=[progress['correct_answers'], progress['incorrect_answers']],
                                names=['Correct ✅', 'Incorrect ❌'],
                                title="Answer Distribution",
                                color=['Correct ✅', 'Incorrect ❌'],
                                color_discrete_map={'Correct ✅': '#38ef7d', 'Incorrect ❌': '#ff6b6b'}
                            )
                            fig_pie.update_traces(textinfo='percent+label')
                            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#e0e0e0"}, height=300)
                            st.plotly_chart(fig_pie, use_container_width=True)
                
                if st.button("🔄 Reset Progress", type="secondary", use_container_width=True):
                    st.session_state.quiz_progress = {
                        'total_questions': 0, 'correct_answers': 0, 'incorrect_answers': 0,
                        'quizzes_taken': 0, 'flashcards_studied': 0
                    }
                    st.rerun()

        else:
            st.error("❌ Could not extract enough text from this PDF. Please try a different document.")

    else:
        st.info("👆 Please upload a PDF document to get started!")
        st.markdown("""
        <div style='background: rgba(56, 239, 125, 0.05); border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 16px; padding: 40px; text-align: center; margin: 20px 0;'>
            <h2 style='color: #38ef7d; margin-bottom: 20px;'>🌿 Welcome to Quillium!</h2>
            <p style='color: #ccc; font-size: 1.1em; line-height: 1.6;'>
            Upload a PDF to create interactive quizzes and flashcards in your preferred language.
            <br>Supports <strong>50+ languages</strong> including Spanish, French, German, Chinese, Arabic, and many more!
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()