import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from pathlib import Path
from data.schemas import AuditLog
from config import get_settings

class AuditTrail:
    def __init__(self, log_file: Optional[str] = None):
        self.settings = get_settings()
        self.logs: List[AuditLog] = []
        if log_file:
            self.log_file = Path(log_file)
        else:
            self.log_file = Path(self.settings.AUDIT_LOG_PATH)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Load existing logs from file on init
        self._load_logs_from_file()
        
    def _load_logs_from_file(self):
        """Load existing audit logs from file on startup."""
        if not self.log_file.exists():
            return
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        log = AuditLog(
                            log_id=data.get('log_id', ''),
                            merchant_id=data.get('merchant_id', ''),
                            action_id=data.get('action_id'),
                            event_type=data.get('event_type', ''),
                            details=data.get('details', {}),
                            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                            severity=data.get('severity', 'info'),
                        )
                        self.logs.append(log)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            pass

    def log_event(
        self,
        merchant_id: str,
        event_type: str,
        details: Dict[str, Any],
        action_id: Optional[str] = None,
        severity: str = "info"
    ) -> AuditLog:
        """Log an audit event."""
        log_entry = AuditLog(
            log_id=str(uuid.uuid4()),
            merchant_id=merchant_id,
            action_id=action_id,
            event_type=event_type,
            details=details,
            timestamp=datetime.now(),
            severity=severity
        )
        
        # Store in memory
        self.logs.append(log_entry)
        
        # Append to file
        self._write_to_file(log_entry)
        
        return log_entry
    
    def _write_to_file(self, log_entry: AuditLog):
        """Write log entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                log_data = {
                    "log_id": log_entry.log_id,
                    "merchant_id": log_entry.merchant_id,
                    "action_id": log_entry.action_id,
                    "event_type": log_entry.event_type,
                    "details": log_entry.details,
                    "timestamp": log_entry.timestamp.isoformat(),
                    "severity": log_entry.severity
                }
                f.write(json.dumps(log_data) + '\n')
        except Exception as e:
            print(f"Failed to write audit log: {e}")
    
    def get_merchant_logs(
        self, 
        merchant_id: str, 
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[AuditLog]:
        """Get audit logs for a specific merchant."""
        filtered_logs = [
            log for log in self.logs
            if log.merchant_id == merchant_id
        ]
        
        if event_type:
            filtered_logs = [
                log for log in filtered_logs
                if log.event_type == event_type
            ]
        
        # Sort by timestamp descending
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_logs[:limit]
    
    def get_action_logs(self, action_id: str) -> List[AuditLog]:
        """Get all logs for a specific action."""
        return [
            log for log in self.logs
            if log.action_id == action_id
        ]
    
    def get_recent_logs(self, limit: int = 50) -> List[AuditLog]:
        """Get recent audit logs across all merchants."""
        sorted_logs = sorted(self.logs, key=lambda x: x.timestamp, reverse=True)
        return sorted_logs[:limit]
    
    def get_error_logs(self, limit: int = 50) -> List[AuditLog]:
        """Get recent error logs."""
        error_logs = [
            log for log in self.logs
            if log.severity in ["error", "warning"]
        ]
        
        sorted_logs = sorted(error_logs, key=lambda x: x.timestamp, reverse=True)
        return sorted_logs[:limit]
    
    def get_merchant_summary(self, merchant_id: str) -> Dict[str, Any]:
        """Get summary of audit events for a merchant."""
        merchant_logs = self.get_merchant_logs(merchant_id, limit=1000)
        
        if not merchant_logs:
            return {"merchant_id": merchant_id, "total_events": 0}
        
        # Count events by type
        event_counts = {}
        for log in merchant_logs:
            event_counts[log.event_type] = event_counts.get(log.event_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for log in merchant_logs:
            severity_counts[log.severity] = severity_counts.get(log.severity, 0) + 1
        
        # Get first and last event
        timestamps = [log.timestamp for log in merchant_logs]
        
        return {
            "merchant_id": merchant_id,
            "total_events": len(merchant_logs),
            "event_type_counts": event_counts,
            "severity_counts": severity_counts,
            "first_event": min(timestamps).isoformat(),
            "last_event": max(timestamps).isoformat(),
            "date_range_days": (max(timestamps) - min(timestamps)).days
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide audit summary."""
        if not self.logs:
            return {"total_events": 0}
        
        # Count by event type
        event_counts = {}
        for log in self.logs:
            event_counts[log.event_type] = event_counts.get(log.event_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for log in self.logs:
            severity_counts[log.severity] = severity_counts.get(log.severity, 0) + 1
        
        # Count unique merchants
        unique_merchants = set(log.merchant_id for log in self.logs)
        
        # Get time range
        timestamps = [log.timestamp for log in self.logs]
        
        return {
            "total_events": len(self.logs),
            "unique_merchants": len(unique_merchants),
            "event_type_counts": event_counts,
            "severity_counts": severity_counts,
            "first_event": min(timestamps).isoformat(),
            "last_event": max(timestamps).isoformat(),
            "average_events_per_merchant": len(self.logs) / len(unique_merchants) if unique_merchants else 0
        }
    
    def export_logs(self, format: str = "json") -> str:
        """Export all logs in specified format."""
        if format == "json":
            logs_data = []
            for log in self.logs:
                logs_data.append({
                    "log_id": log.log_id,
                    "merchant_id": log.merchant_id,
                    "action_id": log.action_id,
                    "event_type": log.event_type,
                    "details": log.details,
                    "timestamp": log.timestamp.isoformat(),
                    "severity": log.severity
                })
            return json.dumps(logs_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_old_logs(self, days_to_keep: int = 30):
        """Clear logs older than specified days."""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        original_count = len(self.logs)
        self.logs = [
            log for log in self.logs
            if log.timestamp.timestamp() > cutoff_date
        ]
        
        cleared_count = original_count - len(self.logs)
        print(f"Cleared {cleared_count} logs older than {days_to_keep} days")
        
        # Rewrite log file
        self.log_file.write_text('')
        for log in self.logs:
            self._write_to_file(log)