function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
}

(function initDetectionUI(){
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const detectBtn = document.getElementById("detectBtn");
  const clearBtn = document.getElementById("clearBtn");
  const statusText = document.getElementById("statusText");

  // If not on home page, exit safely:
  if (!dropZone || !fileInput || !detectBtn) return;

  const originalImg = document.getElementById("originalImg");
  const annotatedImg = document.getElementById("annotatedImg");
  const originalPlaceholder = document.getElementById("originalPlaceholder");
  const annotatedPlaceholder = document.getElementById("annotatedPlaceholder");

  const table = document.getElementById("detectionsTable");
  const tableEmpty = document.getElementById("tableEmpty");
  const totalDetections = document.getElementById("totalDetections");
  const topConfidence = document.getElementById("topConfidence");

  let currentFile = null;

  function setStatus(msg){ if (statusText) statusText.textContent = msg || ""; }

  function resetUI(){
    currentFile = null;
    fileInput.value = "";

    originalImg.style.display = "none";
    annotatedImg.style.display = "none";
    originalImg.src = "";
    annotatedImg.src = "";

    originalPlaceholder.style.display = "grid";
    annotatedPlaceholder.style.display = "grid";
    annotatedPlaceholder.textContent = "Run detection to see results";

    detectBtn.disabled = true;
    if (clearBtn) clearBtn.disabled = true;

    totalDetections.textContent = "0";
    topConfidence.textContent = "—";

    table.innerHTML = "";
    table.appendChild(tableEmpty);
    tableEmpty.style.display = "block";
    setStatus("");
  }

  function showOriginalPreview(file){
    const reader = new FileReader();
    reader.onload = (e) => {
      originalImg.src = e.target.result;
      originalImg.style.display = "block";
      originalPlaceholder.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  function renderTable(dets){
    table.innerHTML = "";

    const head = document.createElement("div");
    head.className = "row head";
    head.innerHTML = `<div>Class</div><div>Confidence</div><div>Box (x1,y1,x2,y2)</div>`;
    table.appendChild(head);

    if (!dets || dets.length === 0){
      table.appendChild(tableEmpty);
      tableEmpty.style.display = "block";
      return;
    }

    dets.forEach((d) => {
      const row = document.createElement("div");
      row.className = "row";
      const confPct = (d.confidence * 100).toFixed(1) + "%";
      row.innerHTML = `
        <div>${d.class}</div>
        <div><span class="badge">${confPct}</span></div>
        <div class="mono">[${d.bbox.join(", ")}]</div>
      `;
      table.appendChild(row);
    });
  }

  function computeTopConfidence(dets){
    if (!dets || dets.length === 0) return null;
    return Math.max(...dets.map(d => d.confidence));
  }

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
  });

  function handleFile(file){
    if (!file.type.startsWith("image/")){
      alert("Please upload an image file.");
      return;
    }
    currentFile = file;
    showOriginalPreview(file);

    annotatedImg.style.display = "none";
    annotatedImg.src = "";
    annotatedPlaceholder.style.display = "grid";
    annotatedPlaceholder.textContent = "Ready. Click Run Detection.";

    detectBtn.disabled = false;
    if (clearBtn) clearBtn.disabled = false;
    setStatus(`Selected: ${file.name}`);
  }

  if (clearBtn) clearBtn.addEventListener("click", resetUI);

  detectBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    detectBtn.disabled = true;
    setStatus("Running detection…");

    const formData = new FormData();
    formData.append("image", currentFile);

    try{const res = await fetch("/api/detect/", {
      method: "POST",
      headers: { "X-CSRFToken": getCookie("csrftoken") },
      body: formData
    });
    // Read JSON safely (401 response must also be parsed)
    let data = {};
    try {
      data = await res.json();
    } catch (e) {
      data = {};
    }

    // ✅ Handle login-required redirect
    if (res.status === 401) {
      const loginUrl = data.login_url || "/accounts/login/";
      const next = data.next || "/";
      window.location.href = `${loginUrl}?next=${encodeURIComponent(next)}`;
      return; // stop further processing
      }
      if (!res.ok) {
        throw new Error(data.error || "Server error");
      }

      const dets = data.detections || [];
      renderTable(dets);

      totalDetections.textContent = String(dets.length);

      const top = computeTopConfidence(dets);
      topConfidence.textContent = top === null ? "—" : (top * 100).toFixed(1) + "%";

      if (data.annotated_image_base64){
        annotatedImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
        annotatedImg.style.display = "block";
        annotatedPlaceholder.style.display = "none";
      }

      setStatus(`Done. Found ${dets.length} object(s).`);
    } catch(err){
      setStatus("");
      alert("Detection failed: " + err.message);
    } finally{
      detectBtn.disabled = false;
    }
  });

  resetUI();
})();
// Scroll Reveal for Developer Section
document.addEventListener('DOMContentLoaded', () => {
  const reveals = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  reveals.forEach(el => observer.observe(el));
});