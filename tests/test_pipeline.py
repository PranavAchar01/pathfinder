import base64
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.pipeline import NavigationPipeline  # noqa: E402
from app.reasoning.conversation import ConversationAgent  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402


def _frame() -> str:
    img = Image.new("RGB", (320, 240), (40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_pipeline_produces_scene():
    scene = NavigationPipeline().process(_frame(), frame_id=1)
    assert scene.risk in {"clear", "caution", "stop"}
    assert scene.narration
    for o in scene.objects:
        assert o.side in {"left", "center", "right"}
        assert 9 <= o.clock <= 12 or 1 <= o.clock <= 3
        assert o.distance_m > 0


def test_haptic_matches_risk():
    scene = NavigationPipeline().process(_frame(), frame_id=2)
    peak = max(scene.haptic.left, scene.haptic.center, scene.haptic.right)
    if scene.risk == "clear":
        assert peak == 0
    else:
        assert peak > 0


def test_conversation_runs():
    resp = ConversationAgent().respond(ChatRequest(text="what is around me?"))
    assert resp.reply
