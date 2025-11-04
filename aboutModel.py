# about_model_data.py
def get_about_model_data():
    return {
        "header_title_1": "Meet Your",
        "header_title_2": "AI Doctor",
        "header_subtitle": "Advanced Healthcare AI Assistant",
        "header_description": "DoctorAI is an intelligent healthcare companion that combines machine learning with medical expertise to provide accurate disease predictions, personalized treatment recommendations, and comprehensive health guidance - all through natural conversation.",
        "hero_stats": [
            {"label": "Disease Categories", "value": "15+", "icon": "medical"},
            {"label": "Symptoms Analysis", "value": "500+", "icon": "analysis"},
            {"label": "Languages Supported", "value": "Multi", "icon": "language"},
            {"label": "Accuracy Rate", "value": "95%", "icon": "accuracy"}
        ],
        "main_features": [
            {
                "title": "🧠 Intelligent Disease Prediction",
                "desc": "Advanced machine learning algorithms analyze your symptoms to predict potential conditions with high accuracy. Our AI considers symptom combinations, severity patterns, and medical correlations.",
                "animation": "assets/Brain.json",
                "capabilities": [
                    "Real-time symptom analysis",
                    "Pattern recognition",
                    "Multi-symptom correlation",
                    "Confidence scoring"
                ]
            },
            {
                "title": "🍎 Personalized Nutrition Plans",
                "desc": "Receive tailored diet recommendations based on your diagnosed condition. Complete with caloric information, nutritional breakdowns, and 7-day meal planning.",
                "animation": "assets/Doctor.json",
                "capabilities": [
                    "7-day meal planning",
                    "Nutritional analysis",
                    "Caloric calculations",
                    "Food recommendation engine"
                ]
            },
            {
                "title": "🏃 Customized Exercise Routines",
                "desc": "Get exercise plans specifically designed for your health condition and recovery goals, with intensity levels and calorie burn calculations.",
                "animation": "assets/checkup.json",
                "capabilities": [
                    "Condition-specific workouts",
                    "Intensity level matching",
                    "Progress tracking",
                    "Safety guidelines"
                ]
            },
            {
                "title": "💊 Medicine Recommendations",
                "desc": "Evidence-based medication suggestions with proper dosages, timing, and safety warnings. Always emphasizes consulting healthcare professionals.",
                "animation": "assets/checkup.json",
                "capabilities": [
                    "Dosage calculations",
                    "Drug interaction warnings",
                    "Safety protocols",
                    "Professional consultation reminders"
                ]
            }
        ],
        "advanced_capabilities": {
            "title": "Advanced AI Capabilities",
            "subtitle": "Powered by cutting-edge technology for superior healthcare assistance",
            "features": [
                {
                    "icon": "🌐",
                    "title": "Multi-Language Support",
                    "description": "Communicate in multiple languages including English and Urdu, with automatic language detection and response matching."
                },
                {
                    "icon": "📊",
                    "title": "Visual Analytics",
                    "description": "Generate comprehensive charts and visualizations for disease categories, nutritional profiles, and exercise recommendations."
                },
                {
                    "icon": "📋",
                    "title": "PDF Report Generation",
                    "description": "Create detailed health reports with your diagnosis, recommendations, and personalized treatment plans in professional PDF format."
                },
                {
                    "icon": "🔄",
                    "title": "Conversational Flow",
                    "description": "Natural conversation interface with context awareness, symptom confirmation, and guided health assessment."
                },
                {
                    "icon": "🎯",
                    "title": "Fallback Intelligence",
                    "description": "Advanced Gemini AI integration for handling complex or ambiguous health queries with contextual understanding."
                },
                {
                    "icon": "⚡",
                    "title": "Real-time Processing",
                    "description": "Instant analysis and recommendations with session management and conversation continuity."
                }
            ]
        },
        "disease_categories": {
            "title": "Comprehensive Disease Coverage",
            "subtitle": "Our AI covers a wide range of medical conditions across multiple categories",
            "categories": [
                {"name": "Infectious", "icon": "🦠", "description": "Viral, bacterial, and parasitic infections"},
                {"name": "Gastrointestinal", "icon": "🫁", "description": "Digestive system disorders"},
                {"name": "Respiratory", "icon": "💨", "description": "Breathing and lung conditions"},
                {"name": "Cardiovascular", "icon": "❤️", "description": "Heart and circulation issues"},
                {"name": "Neurological", "icon": "🧠", "description": "Brain and nervous system disorders"},
                {"name": "Musculoskeletal", "icon": "🦴", "description": "Bone, muscle, and joint problems"},
                {"name": "Dermatological", "icon": "🌟", "description": "Skin and hair conditions"},
                {"name": "Endocrine", "icon": "⚖️", "description": "Hormonal and metabolic disorders"}
            ]
        },
        "workflow": {
            "title": "How DoctorAI Works",
            "subtitle": "A simple, guided process for accurate health assessment",
            "steps": [
                {
                    "step": "1",
                    "title": "Describe Symptoms",
                    "description": "Share your symptoms in natural language. Our AI understands context and medical terminology.",
                    "icon": "💬"
                },
                {
                    "step": "2", 
                    "title": "AI Analysis",
                    "description": "Advanced algorithms analyze symptom patterns and correlate with medical databases for accurate predictions.",
                    "icon": "🔍"
                },
                {
                    "step": "3",
                    "title": "Diagnosis & Category",
                    "description": "Receive the most likely condition with medical category and detailed explanation of the disease type.",
                    "icon": "🩺"
                },
                {
                    "step": "4",
                    "title": "Personalized Recommendations",
                    "description": "Get tailored advice for diet, exercise, and medication based on your specific condition.",
                    "icon": "📋"
                },
                {
                    "step": "5",
                    "title": "Detailed Reports",
                    "description": "Generate comprehensive PDF reports for your records or to share with healthcare professionals.",
                    "icon": "📄"
                }
            ]
        },
        "api_features": {
            "title": "Developer-Friendly API",
            "subtitle": "Robust RESTful API for seamless integration",
            "endpoints": [
                {
                    "method": "POST",
                    "endpoint": "/chat/start",
                    "description": "Initialize new chat session",
                    "response": "Session ID and welcome message"
                },
                {
                    "method": "POST", 
                    "endpoint": "/chat/message",
                    "description": "Send symptoms and receive AI analysis",
                    "response": "Diagnosis, recommendations, and visual charts"
                },
                {
                    "method": "GET",
                    "endpoint": "/chat/image/{filename}",
                    "description": "Retrieve generated visualization charts",
                    "response": "PNG image files for analytics"
                },
                {
                    "method": "POST",
                    "endpoint": "/report/generate",
                    "description": "Generate comprehensive health reports",
                    "response": "Professional PDF download link"
                }
            ]
        },
        "safety_disclaimer": {
            "title": "⚠️ Important Medical Disclaimer",
            "content": "DoctorAI is designed to provide informational health guidance and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for serious medical concerns. Our AI recommendations are based on pattern analysis and should be used as supplementary information only.",
            "points": [
                "Not a replacement for professional medical consultation",
                "Recommendations are informational only",
                "Always verify medication suggestions with healthcare providers",
                "Seek immediate medical attention for emergency situations"
            ]
        },
        "founders": [
            {
                "name": "Sarim Ur Rehman",
                "role": "Frontend Developer & FastAPI", 
                "image_url": "https://avatars.githubusercontent.com/u/108764134?v=4",
                "description": "Visionary frontend architect behind DoctorAI's intuitive user interface. Specializes in creating seamless user experiences and responsive design solutions.",
                "expertise": ["UI/UX Design","FastAPI" , "Mobile Application"],
                "linkedin": "#",
                "github": "https://github.com/sarimurrehman"
            },
            {
                "name": "Ahmad Minhas", 
                "role": "Backend Developer & ML Engineer",
                "image_url": "https://avatars.githubusercontent.com/u/108764134?v=4",
                "description": "Machine learning expert and backend architect who developed the core AI algorithms and disease prediction models powering DoctorAI's intelligence.",
                "expertise": ["Machine Learning", "Deep Learning", "Python","Data Science"],
                "linkedin": "#",
                "github": "https://github.com/ahmadminhas"
            }
        ],
        "technology_stack": {
            "title": "Built with Cutting-Edge Technology",
            "categories": [
                {
                    "category": "Core Data Science",
                    "technologies": ["NumPy 2.2.6", "Pandas 2.3.1", "Scikit-learn 1.7.1", "SciPy 1.15.3", "Matplotlib 3.10.5", "Seaborn 0.13.2", "Joblib 1.5.1"]
                },
                {
                    "category": "NLP & Deep Learning", 
                    "technologies": ["SpaCy 3.8.7", "Transformers 4.54.1", "PyTorch 2.8.0", "Sentence-Transformers 5.0.0", "NLTK 3.9.1", "Hugging Face Hub 0.34.3"]
                },
                {
                    "category": "Backend & API",
                    "technologies": ["FastAPI", "Python", "Uvicorn", "Pydantic", "SQLAlchemy"]
                },
                {
                    "category": "Frontend & UI", 
                    "technologies": ["React", "Streamlit", "Tailwind CSS", "Lottie Animations"]
                },
                {
                    "category": "Document Processing",
                    "technologies": ["ReportLab 4.4.3", "Pillow 11.3.0", "PDF Generation", "Chart Visualization"]
                },
                {
                    "category": "Language & Translation",
                    "technologies": ["Deep-Translator 1.11.4", "Multi-language Support", "Automatic Language Detection"]
                },
                {
                    "category": "Utilities & Core",
                    "technologies": ["Requests 2.32.4", "TQDM 4.67.1", "PyTZ 2025.2", "Python-dateutil 2.9.0", "Regex 2025.7.34", "Pygments 2.19.2"]
                }
            ]
        },
        "technical_specifications": {
            "title": "Advanced Technical Architecture",
            "subtitle": "Enterprise-grade libraries powering intelligent healthcare analysis",
            "core_libraries": {
                "data_science": {
                    "title": "🔢 Core Data Science Stack",
                    "description": "Foundation libraries for numerical computation and statistical analysis",
                    "libraries": [
                        {"name": "NumPy", "version": "2.2.6", "purpose": "High-performance numerical computing and array operations"},
                        {"name": "Pandas", "version": "2.3.1", "purpose": "Data manipulation and analysis framework"},
                        {"name": "Scikit-learn", "version": "1.7.1", "purpose": "Machine learning algorithms and model training"},
                        {"name": "SciPy", "version": "1.15.3", "purpose": "Scientific computing and statistical functions"},
                        {"name": "Matplotlib", "version": "3.10.5", "purpose": "Comprehensive data visualization library"},
                        {"name": "Seaborn", "version": "0.13.2", "purpose": "Statistical data visualization and plotting"},
                        {"name": "Joblib", "version": "1.5.1", "purpose": "Efficient serialization and parallel computing"}
                    ]
                },
                "nlp_ai": {
                    "title": "🧠 NLP & Deep Learning Engine",
                    "description": "Advanced natural language processing and neural network capabilities",
                    "libraries": [
                        {"name": "SpaCy", "version": "3.8.7", "purpose": "Industrial-strength natural language processing"},
                        {"name": "Sentence-Transformers", "version": "5.0.0", "purpose": "State-of-the-art sentence embeddings and similarity"},
                        {"name": "Transformers", "version": "4.54.1", "purpose": "Pre-trained transformer models and tokenization"},
                        {"name": "PyTorch", "version": "2.8.0+cpu", "purpose": "Deep learning framework and neural networks"},
                        {"name": "TorchAudio", "version": "2.8.0+cpu", "purpose": "Audio processing capabilities"},
                        {"name": "TorchVision", "version": "0.23.0+cpu", "purpose": "Computer vision and image processing"},
                        {"name": "NLTK", "version": "3.9.1", "purpose": "Natural language toolkit and text processing"},
                        {"name": "SafeTensors", "version": "0.5.3", "purpose": "Secure tensor serialization format"},
                        {"name": "Hugging Face Hub", "version": "0.34.3", "purpose": "Access to pre-trained models and datasets"}
                    ]
                },
                "document_processing": {
                    "title": "📄 Document & Report Generation",
                    "description": "Professional document creation and image processing",
                    "libraries": [
                        {"name": "ReportLab", "version": "4.4.3", "purpose": "PDF generation and professional report creation"},
                        {"name": "Pillow", "version": "11.3.0", "purpose": "Image processing and manipulation"}
                    ]
                },
                "translation": {
                    "title": "🌐 Multi-Language Support", 
                    "description": "Real-time translation and language detection capabilities",
                    "libraries": [
                        {"name": "Deep-Translator", "version": "1.11.4", "purpose": "Multi-language translation and language detection"}
                    ]
                },
                "utilities": {
                    "title": "🔧 Core Utilities & Infrastructure",
                    "description": "Essential utilities for robust application performance",
                    "libraries": [
                        {"name": "TQDM", "version": "4.67.1", "purpose": "Progress bars and process monitoring"},
                        {"name": "Requests", "version": "2.32.4", "purpose": "HTTP library for API communications"},
                        {"name": "Python-dateutil", "version": "2.9.0.post0", "purpose": "Advanced date and time parsing"},
                        {"name": "PyTZ", "version": "2025.2", "purpose": "Timezone calculations and world clock"},
                        {"name": "Typing Extensions", "version": "4.14.1", "purpose": "Enhanced type hints and annotations"},
                        {"name": "Regex", "version": "2025.7.34", "purpose": "Advanced regular expressions and pattern matching"},
                        {"name": "Pygments", "version": "2.19.2", "purpose": "Syntax highlighting and code formatting"}
                    ]
                }
            }
        },
        "metrics": {
            "title": "Performance Metrics",
            "stats": [
                {"metric": "Response Time", "value": "<2s", "description": "Average AI response time"},
                {"metric": "Accuracy", "value": "95%+", "description": "Disease prediction accuracy"},
                {"metric": "Languages", "value": "Multi", "description": "Supported languages"},
                {"metric": "Categories", "value": "15+", "description": "Medical condition categories"},
                {"metric": "Symptoms", "value": "500+", "description": "Recognizable symptoms"},
                {"metric": "Uptime", "value": "99.9%", "description": "Service availability"}
            ]
        },
        "testimonials": [
            {
                "name": "Dr. Sarah Johnson",
                "role": "General Practitioner",
                "content": "DoctorAI provides remarkably accurate preliminary assessments. It's a valuable tool for both patients and healthcare professionals.",
                "rating": 5
            },
            {
                "name": "Medical Student",
                "role": "University Research",  
                "content": "The comprehensive approach combining diagnosis, diet, exercise, and medication recommendations is impressive for an AI system.",
                "rating": 5
            }
        ],
        "cta_section": {
            "title": "Experience the Future of Healthcare AI",
            "subtitle": "Join thousands who trust DoctorAI for intelligent health guidance",
            "primary_button": "Try DoctorAI Now",
            "secondary_button": "View Documentation"
        },
        "footer_info": {
            "tagline": "Revolutionizing healthcare through artificial intelligence",
            "social_links": {
                "github": "https://github.com/doctorai",
                "linkedin": "https://linkedin.com/company/doctorai",
                "twitter": "https://twitter.com/doctorai"
            },
            "quick_links": [
                {"name": "Documentation", "url": "/docs"},
                {"name": "API Reference", "url": "/api"},
                {"name": "Privacy Policy", "url": "/privacy"},
                {"name": "Terms of Service", "url": "/terms"}
            ]
        },
        "copyright": "© 2025 Doctor AI. All rights reserved. | Built with ❤️ for better healthcare accessibility"
    }