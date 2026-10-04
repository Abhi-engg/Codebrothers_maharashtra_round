// Re:Learn Frontend Application Logic

const API_BASE_URL = 'http://localhost:8000/api';
let currentStudentId = 1; // Demo student
let currentQuestion = null;
let currentDiagnosis = null;
let currentIntervention = null;
let whiteboard = null;

const app = {
  init() {
    this.bindNavigation();
    
    // Initialize Whiteboard if available
    if (typeof ReLearnWhiteboard !== 'undefined') {
      whiteboard = new ReLearnWhiteboard('whiteboard-canvas');
      
      document.getElementById('wb-play').addEventListener('click', () => whiteboard.play());
      document.getElementById('wb-pause').addEventListener('click', () => whiteboard.pause());
      document.getElementById('wb-reset').addEventListener('click', () => whiteboard.reset());
    }

    // Load initial data
    this.loadProfile();
  },

  bindNavigation() {
    document.querySelectorAll('.nav-links li').forEach(link => {
      link.addEventListener('click', (e) => {
        // Remove active class from all links
        document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
        // Add to clicked
        e.currentTarget.classList.add('active');
        
        // Show corresponding view
        const viewId = e.currentTarget.getAttribute('data-view');
        this.showView(viewId);
      });
    });
  },

  showView(viewId) {
    document.querySelectorAll('.view').forEach(view => {
      view.classList.remove('active');
    });
    document.getElementById(viewId).classList.add('active');
  },

  async loadProfile() {
    try {
      const response = await fetch(`${API_BASE_URL}/learner/${currentStudentId}/profile`);
      if (response.ok) {
        const data = await response.json();
        // Update UI if needed. For now, we keep the static demo UI in Home
        console.log("Profile loaded:", data);
      }
    } catch (err) {
      console.error("Error loading profile:", err);
    }
  },

  async loadQuestion() {
    this.showView('view-question');
    document.getElementById('q-title').innerText = "Loading...";
    document.getElementById('q-body').innerHTML = "";
    document.getElementById('q-answer').value = "";
    document.getElementById('q-working').value = "";

    try {
      // Fetch available questions
      const response = await fetch(`${API_BASE_URL}/quiz/questions`);
      if (!response.ok) throw new Error("Failed to load questions");
      const questions = await response.json();
      
      // Pick the first one for the demo
      if (questions.length > 0) {
        currentQuestion = questions[0];
        document.getElementById('q-title').innerText = currentQuestion.template_id;
        document.getElementById('q-body').innerText = currentQuestion.question_text;
      }
    } catch (err) {
      console.error(err);
      document.getElementById('q-title').innerText = "Error loading question";
    }
  },

  async submitAnswer() {
    const answer = document.getElementById('q-answer').value;
    const working = document.getElementById('q-working').value;
    
    if (!answer || !currentQuestion) return;

    const btn = document.getElementById('btn-submit-answer');
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader"></i> Diagnosing...`;
    lucide.createIcons();

    try {
      // Call diagnose endpoint
      const response = await fetch(`${API_BASE_URL}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: currentStudentId,
          question_text: currentQuestion.question_text,
          correct_solution: "N/A", // Ideally fetched from question
          student_answer: answer,
          student_working: working,
          image_data: null,
          use_sequence_model: true
        })
      });

      if (!response.ok) throw new Error("Diagnosis failed");
      currentDiagnosis = await response.json();
      
      this.renderDiagnosis(currentDiagnosis);
      
    } catch (err) {
      console.error(err);
      alert("Error diagnosing answer. See console.");
    } finally {
      btn.disabled = false;
      btn.innerText = "Submit";
    }
  },

  renderDiagnosis(diagnosis) {
    this.showView('view-diagnosis');
    
    // Update diagnosis UI
    document.getElementById('diag-name').innerText = diagnosis.misconception_id !== 'UNCERTAIN' 
      ? `Possible misconception: ${diagnosis.misconception_id}` 
      : 'Uncertain';
      
    document.getElementById('diag-conf').innerText = `${(diagnosis.confidence * 100).toFixed(0)}%`;
    document.getElementById('diag-evidence').innerText = `Model identified this pattern based on your steps.`;
  },

  async showIntervention() {
    if (!currentDiagnosis || currentDiagnosis.misconception_id === 'UNCERTAIN') {
       // Skip to reassessment if uncertain
       this.showReassessment();
       return;
    }

    this.showView('view-intervention');
    document.getElementById('interv-explanation').innerText = "Generating intervention...";

    try {
      const response = await fetch(`${API_BASE_URL}/intervention`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: currentStudentId,
          misconception_id: currentDiagnosis.misconception_id,
          original_question: currentQuestion.question_text,
          student_working: document.getElementById('q-working').value
        })
      });
      
      if (!response.ok) throw new Error("Intervention failed");
      currentIntervention = await response.json();
      
      document.getElementById('interv-explanation').innerText = currentIntervention.explanation;
      
      if (whiteboard && currentIntervention.whiteboard_commands) {
         whiteboard.loadSequence(currentIntervention.whiteboard_commands);
         whiteboard.play();
      }

    } catch (err) {
      console.error(err);
      document.getElementById('interv-explanation').innerText = "Failed to generate visual explanation.";
    }
  },

  async showReassessment() {
    this.showView('view-reassessment');
    document.getElementById('re-title').innerText = "Transfer Question";
    // For demo purposes, hardcoding a transfer question text.
    document.getElementById('re-body').innerText = "A 12V battery is connected to a 3Ω resistor. What happens to the current if a second identical resistor is added in series?";
    document.getElementById('re-answer').value = "";
    document.getElementById('re-working').value = "";
  },

  async submitReassessment() {
    const answer = document.getElementById('re-answer').value;
    const working = document.getElementById('re-working').value;
    
    if (!answer) return;

    const btn = document.getElementById('btn-submit-re');
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader"></i> Updating...`;
    lucide.createIcons();

    try {
      const response = await fetch(`${API_BASE_URL}/reassessment/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: currentStudentId,
          misconception_id: currentDiagnosis ? currentDiagnosis.misconception_id : "MISC-UNKNOWN",
          transfer_question: document.getElementById('re-body').innerText,
          student_answer: answer,
          student_working: working
        })
      });

      if (!response.ok) throw new Error("Reassessment failed");
      const result = await response.json();
      
      alert(`Mastery Status Updated to: ${result.new_status}`);
      
      // Go back to home and reload profile
      this.showView('view-home');
      this.loadProfile();

    } catch (err) {
      console.error(err);
      alert("Error submitting reassessment. See console.");
    } finally {
      btn.disabled = false;
      btn.innerText = "Submit";
    }
  }
};

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  app.init();
});
