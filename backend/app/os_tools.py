import os
import shutil
import subprocess
import uuid
from typing import Dict, Any, Optional, List

BACKUP_DIR = os.path.join(os.path.expanduser("~"), ".local_agent_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# In-memory session tracking
PENDING_ACTIONS: Dict[str, Dict[str, Any]] = {}
UNDO_STACK: List[Dict[str, Any]] = []


def propose_action(
    action_type: str, 
    target: str, 
    payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Stage 1: Generates an action proposal that requires user confirmation."""
    action_id = str(uuid.uuid4())[:8]
    proposal: Dict[str, Any] = {
        "action_id": action_id,
        "type": action_type,
        "target": target,
        "payload": payload or {},
        "description": f"AI requests permission to: {action_type} on '{target}'",
        "status": "pending",
    }
    PENDING_ACTIONS[action_id] = proposal
    return proposal


def execute_approved_action(action_id: str) -> Dict[str, Any]:
    """Stage 2: Runs strictly after user approves."""
    action = PENDING_ACTIONS.pop(action_id, None)
    if not action:
        return {"status": "error", "message": "Action not found or expired"}

    action_type = action["type"]
    target = action["target"]
    payload = action["payload"]

    try:
        if action_type == "delete_file":
            if not os.path.exists(target):
                return {"status": "error", "message": "Target file does not exist"}
            backup_path = os.path.join(
                BACKUP_DIR, f"{os.path.basename(target)}.{uuid.uuid4().hex[:6]}.bak"
            )
            shutil.copy2(target, backup_path)
            os.remove(target)
            UNDO_STACK.append(
                {"undo_type": "restore_file", "backup": backup_path, "target": target}
            )
            return {"status": "success", "message": f"Deleted {target}. (Undo available)"}

        elif action_type == "create_file":
            content = payload.get("content", "")
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            UNDO_STACK.append({"undo_type": "delete_file", "target": target})
            return {"status": "success", "message": f"Created file at {target}. (Undo available)"}

        elif action_type == "launch_app":
            proc = subprocess.Popen(target, shell=True)
            UNDO_STACK.append(
                {"undo_type": "kill_process", "pid": proc.pid, "target": target}
            )
            return {
                "status": "success",
                "message": f"Application '{target}' launched. (PID: {proc.pid})",
            }

        elif action_type == "create_folder":
            os.makedirs(target, exist_ok=True)
            UNDO_STACK.append({"undo_type": "remove_folder", "target": target})
            return {"status": "success", "message": f"Created folder {target}."}

        return {"status": "error", "message": f"Unsupported action type: {action_type}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def undo_last_action() -> Dict[str, Any]:
    """Rolls back the most recent executed operation."""
    if not UNDO_STACK:
        return {"status": "error", "message": "Nothing to undo"}

    undo_item = UNDO_STACK.pop()
    undo_type = undo_item["undo_type"]

    try:
        if undo_type == "restore_file":
            shutil.move(undo_item["backup"], undo_item["target"])
            return {"status": "success", "message": f"Restored: {undo_item['target']}"}

        elif undo_type == "delete_file":
            if os.path.exists(undo_item["target"]):
                os.remove(undo_item["target"])
            return {
                "status": "success",
                "message": f"Removed created file: {undo_item['target']}",
            }

        elif undo_type == "remove_folder":
            if os.path.exists(undo_item["target"]):
                os.rmdir(undo_item["target"])
            return {
                "status": "success",
                "message": f"Removed directory: {undo_item['target']}",
            }

        elif undo_type == "kill_process":
            import psutil

            parent = psutil.Process(undo_item["pid"])
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            return {
                "status": "success",
                "message": f"Terminated app '{undo_item['target']}' (PID: {undo_item['pid']})",
            }

        return {"status": "error", "message": "Unknown undo operation"}
    except Exception as e:
        return {"status": "error", "message": f"Undo failed: {str(e)}"}