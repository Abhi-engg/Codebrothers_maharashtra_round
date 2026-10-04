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

  async startQuiz() {
    this.showView('view-question');
    document.getElementById('q-title').innerText = "Loading quiz...";
    document.getElementById('q-body').innerHTML = "";
    document.getElementById('q-answer').value = "";
    document.getElementById('q-working').value = "";

    try {
      // Fetch available questions
      const response = await fetch(`${API_BASE_URL}/quiz/questions`);
      if (!response.ok) throw new Error("Failed to load questions");
      const data = await response.json();
      
      if (data.questions && data.questions.length > 0) {
        this.quizQuestions = data.questions;
        this.currentQuestionIndex = 0;
        this.renderCurrentQuestion();
      } else {
        document.getElementById('q-title').innerText = "No questions found";
      }
    } catch (err) {
      console.error(err);
      document.getElementById('q-title').innerText = "Error loading quiz";
    }
  },

  renderCurrentQuestion() {
    currentQuestion = this.quizQuestions[this.currentQuestionIndex];
    document.getElementById('q-title').innerText = `Question ${this.currentQuestionIndex + 1} of ${this.quizQuestions.length}`;
    document.getElementById('q-body').innerText = currentQuestion.question_text;
    document.getElementById('q-answer').value = "";
    document.getElementById('q-working').value = "";
  },

  nextQuestion() {
    if (this.currentQuestionIndex < this.quizQuestions.length - 1) {
      this.currentQuestionIndex++;
      this.renderCurrentQuestion();
      this.showView('view-question');
    } else {
      alert("Quiz complete! Returning home.");
      this.showView('view-home');
      this.loadProfile();
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
    
    const isUncertain = !diagnosis || diagnosis.misconception_id === 'UNCERTAIN';
    
    const successBox = document.getElementById('diag-success-box');
    const uncertainBox = document.getElementById('diag-uncertain-box');
    const evidenceContainer = document.getElementById('diag-evidence-container');
    const actionRow = document.getElementById('diag-action-row');

    if (isUncertain) {
      successBox.style.display = 'none';
      evidenceContainer.style.display = 'none';
      actionRow.style.display = 'none';
      uncertainBox.style.display = 'block';
      document.getElementById('diag-followup-answer').value = '';
    } else {
      uncertainBox.style.display = 'none';
      successBox.style.display = 'block';
      evidenceContainer.style.display = 'block';
      actionRow.style.display = 'flex';
      
      document.getElementById('diag-name').innerText = `Misconception detected:\n${diagnosis.misconception_id}`;
      document.getElementById('diag-conf').innerText = `${(diagnosis.confidence * 100).toFixed(0)}%`;
      
      const studentReasoning = document.getElementById('q-working').value || "No reasoning provided.";
      document.getElementById('diag-evidence').innerHTML = `You reasoned: <br><i>"${studentReasoning}"</i><br><br><strong>Why this is incorrect:</strong> This contradicts the established physical relationship.`;
      
      // Technical panel
      document.getElementById('tech-misc-id').innerText = diagnosis.misconception_id;
      document.getElementById('tech-conf').innerText = `${(diagnosis.confidence * 100).toFixed(1)}%`;
      // We don't have exact history from the basic API return without a profile fetch, so we mock it for the demo
      document.getElementById('tech-history').innerText = "3 times (Session 2)";
      document.getElementById('tech-status').innerText = "Unresolved";
    }
  },

  async submitFollowup() {
    const followupAnswer = document.getElementById('diag-followup-answer').value;
    if (!followupAnswer) return;

    const btn = document.getElementById('btn-submit-followup');
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader"></i> Diagnosing...`;
    lucide.createIcons();

    // We combine the new followup with the previous working to re-diagnose
    const combinedWorking = document.getElementById('q-working').value + "\nFollow-up: " + followupAnswer;
    
    try {
      const response = await fetch(`${API_BASE_URL}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: currentStudentId,
          question_text: currentQuestion.question_text,
          correct_solution: "N/A",
          student_answer: document.getElementById('q-answer').value,
          student_working: combinedWorking,
          image_data: null,
          use_sequence_model: true
        })
      });

      if (!response.ok) throw new Error("Diagnosis failed");
      currentDiagnosis = await response.json();
      
      // Re-render diagnosis with new result
      this.renderDiagnosis(currentDiagnosis);
      
    } catch (err) {
      console.error(err);
      alert("Error on follow-up diagnosis. See console.");
    } finally {
      btn.disabled = false;
      btn.innerText = "Submit Follow-up";
    }
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
      
      // Continue to next question in the sequence
      this.nextQuestion();

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
