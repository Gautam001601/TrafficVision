from django.contrib.auth.decorators import login_required
import json
import sys
import base64
from pathlib import Path
from pathlib import Path

# Fix config import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .models import *
import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie
from ultralytics import YOLO
from .models import Contact, DetectionLog
from django.urls import reverse
from config import ARTIFACTS_DIR

def find_model_path():
    saved = ARTIFACTS_DIR / "best_model_path.txt"
    if saved.exists():
        p = Path(saved.read_text().strip())
        if p.exists():
            return p
    fallback = Path(__file__).resolve().parent.parent.parent / "yolov8n.pt"
    return fallback

MODEL_PATH = find_model_path()
model = YOLO(str(MODEL_PATH))


@ensure_csrf_cookie
def index(request):
    return render(request, "index.html", {"active": "home"})


@ensure_csrf_cookie
def about(request):
    return render(request, "about.html", {"active": "about"})


@ensure_csrf_cookie
def contact(request):
    context = {"active": "contact", "sent": False}  # ✅ IMPORTANT

    if request.method == "POST":
        nm = request.POST.get('name', '').strip()
        em = request.POST.get('email', '').strip()
        pnum = request.POST.get('pnumber', '').strip()
        msg = request.POST.get('msg', '').strip()

        if not pnum:
            context["error"] = "Phone number is required!"
            return render(request, "contact.html", context)

        Contact.objects.create(
            name=nm,
            email=em,
            pnumber=pnum,
            message=msg
        )

        context["success"] = "Form submitted successfully!"
        return render(request, "contact.html", context)

    return render(request, "contact.html", context)


@require_POST
def api_detect(request):
    # ✅ Require login
    if not request.user.is_authenticated:
        return JsonResponse({
            "login_required": True,
            "login_url": "/accounts/login/",
            "next": "/",   # go back to home/detection
        }, status=401)

    try:
        if "image" not in request.FILES:
            return JsonResponse({"error": "No image uploaded. Field must be 'image'."}, status=400)

        img_file = request.FILES["image"]
        pil_img = Image.open(img_file).convert("RGB")
        img = np.array(pil_img)

        results = model(img, verbose=False)
        r0 = results[0]

        detections = []
        annotated = img.copy()
        names = getattr(model, "names", {})

        # Draw boxes if present
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                label = names.get(cls_id, str(cls_id))

                detections.append({
                    "class": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (69, 165, 227), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (104, 162, 145),
                    2,
                    cv2.LINE_AA,
                )

        # Always encode and return (even if no detections)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return JsonResponse({"error": "Failed to encode output image."}, status=500)

        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        # Save detection to history
        if request.user.is_authenticated:
            from django.core.files.base import ContentFile
            import io
            result_summary = json.dumps(detections)
            img_file.seek(0)
            log = DetectionLog(user=request.user, result=result_summary)
            log.image.save(img_file.name, ContentFile(img_file.read()), save=True)

        return JsonResponse({
            "detections": detections,
            "annotated_image_base64": annotated_b64
        }, status=200)

    except Exception as e:
        # ✅ This ensures you NEVER get "returned None"
        return JsonResponse({"error": str(e)}, status=500)
    
@login_required
def detection_history(request):
    logs = DetectionLog.objects.filter(
        user=request.user
    ).order_by('-created_at')
    return render(request, 'detect/history.html', {'logs': logs})
