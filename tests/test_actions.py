import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.action_orchestrator import ActionOrchestrator
from services.audit_trail import AuditTrail
from data.schemas import GrowthRecommendation, ActionExecution

class TestActionOrchestrator:
    """Test suite for action orchestration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = ActionOrchestrator()
        self.audit_trail = AuditTrail()
        
        # Sample merchant data
        self.merchant_data = {
            "merchant_id": "M0001",
            "business_name": "Test Shop",
            "total_revenue": 100000,
            "average_order_value": 2000,
            "failure_rate": 0.15,
            "status": "at_risk"
        }
        
        # Sample recommendation
        self.recommendation = GrowthRecommendation(
            recommendation_id="rec_001",
            merchant_id="M0001",
            action_type="discount",
            priority=1,
            expected_impact=15000,
            reasoning="High failure rate indicates need for discount campaign",
            parameters={
                "discount_percentage": 10,
                "duration_days": 14,
                "target_segments": ["inactive"]
            },
            created_at=datetime.now()
        )
    
    def test_action_execution(self):
        """Test single action execution."""
        async def run_test():
            action = await self.orchestrator.execute_recommendation(
                self.recommendation,
                self.merchant_data
            )
            
            assert action is not None
            assert isinstance(action, ActionExecution)
            assert action.merchant_id == "M0001"
            assert action.action_type == "discount"
            assert action.status in ["completed", "failed"]
            assert action.action_id is not None
        
        asyncio.run(run_test())
    
    def test_action_status_tracking(self):
        """Test action status tracking."""
        async def run_test():
            action = await self.orchestrator.execute_recommendation(
                self.recommendation,
                self.merchant_data
            )
            
            # Check status tracking
            retrieved_action = self.orchestrator.get_action_status(action.action_id)
            assert retrieved_action is not None
            assert retrieved_action.action_id == action.action_id
            
            # Check merchant actions
            merchant_actions = self.orchestrator.get_merchant_actions("M0001")
            assert len(merchant_actions) > 0
            assert any(a.action_id == action.action_id for a in merchant_actions)
        
        asyncio.run(run_test())
    
    def test_action_cancellation(self):
        """Test action cancellation."""
        async def run_test():
            # Create a recommendation that will be pending
            recommendation = GrowthRecommendation(
                recommendation_id="rec_cancel",
                merchant_id="M0002",
                action_type="outreach",
                priority=2,
                expected_impact=5000,
                reasoning="Test cancellation",
                parameters={"channels": ["email"]},
                created_at=datetime.now()
            )
            
            action = await self.orchestrator.execute_recommendation(
                recommendation,
                {"merchant_id": "M0002"}
            )
            
            # Try to cancel (may fail if already completed)
            success = self.orchestrator.cancel_action(action.action_id)
            
            # Verify cancellation attempt
            assert isinstance(success, bool)
        
        asyncio.run(run_test())
    
    def test_execution_stats(self):
        """Test execution statistics."""
        async def run_test():
            # Execute some actions
            await self.orchestrator.execute_recommendation(
                self.recommendation,
                self.merchant_data
            )
            
            stats = self.orchestrator.get_execution_stats()
            
            assert 'total_actions' in stats
            assert 'completed' in stats
            assert 'failed' in stats
            assert 'pending' in stats
            assert 'success_rate' in stats
            assert 'active_merchants' in stats
            
            assert stats['total_actions'] > 0
            assert 0 <= stats['success_rate'] <= 1
        
        asyncio.run(run_test())
    
    def test_multiple_recommendation_types(self):
        """Test execution of different recommendation types."""
        async def run_test():
            recommendations = [
                GrowthRecommendation(
                    recommendation_id=f"rec_{i}",
                    merchant_id="M0003",
                    action_type=action_type,
                    priority=1,
                    expected_impact=10000,
                    reasoning=f"Test {action_type}",
                    parameters={"test": True},
                    created_at=datetime.now()
                )
                for i, action_type in enumerate(["discount", "retry", "outreach", "campaign"])
            ]
            
            actions = []
            for rec in recommendations:
                action = await self.orchestrator.execute_recommendation(
                    rec,
                    {"merchant_id": "M0003"}
                )
                actions.append(action)
            
            assert len(actions) == 4
            assert all(isinstance(a, ActionExecution) for a in actions)
            assert len(set(a.action_type for a in actions)) == 4
        
        asyncio.run(run_test())

class TestAuditTrail:
    """Test suite for audit trail."""
    
    def setup_method(self):
        """Setup test fixtures with a temporary log file."""
        import tempfile
        self._tmp_dir = tempfile.mkdtemp()
        self._tmp_log = os.path.join(self._tmp_dir, 'test_audit.log')
        # Create a fresh empty file for this test
        open(self._tmp_log, 'w').close()
        self.audit_trail = AuditTrail(log_file=self._tmp_log)
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
    
    def test_log_event(self):
        """Test event logging."""
        log_entry = self.audit_trail.log_event(
            merchant_id="M0001",
            event_type="test_event",
            details={"test_key": "test_value"},
            severity="info"
        )
        
        assert log_entry is not None
        assert log_entry.merchant_id == "M0001"
        assert log_entry.event_type == "test_event"
        assert log_entry.severity == "info"
        assert log_entry.log_id is not None
    
    def test_get_merchant_logs(self):
        """Test retrieving merchant logs."""
        # Log some events
        for i in range(5):
            self.audit_trail.log_event(
                merchant_id="M0002",
                event_type=f"event_{i}",
                details={"index": i}
            )
        
        logs = self.audit_trail.get_merchant_logs("M0002")
        
        assert len(logs) == 5
        assert all(log.merchant_id == "M0002" for log in logs)
    
    def test_get_action_logs(self):
        """Test retrieving action logs."""
        # Log events with action ID
        action_id = "action_123"
        for i in range(3):
            self.audit_trail.log_event(
                merchant_id="M0003",
                event_type=f"action_event_{i}",
                details={"action_step": i},
                action_id=action_id
            )
        
        logs = self.audit_trail.get_action_logs(action_id)
        
        assert len(logs) == 3
        assert all(log.action_id == action_id for log in logs)
    
    def test_get_merchant_summary(self):
        """Test merchant summary generation."""
        # Log various events
        for i in range(10):
            self.audit_trail.log_event(
                merchant_id="M0004",
                event_type="test_event",
                details={"test": True},
                severity="info" if i < 8 else "error"
            )
        
        summary = self.audit_trail.get_merchant_summary("M0004")
        
        assert summary['total_events'] == 10
        assert 'event_type_counts' in summary
        assert 'severity_counts' in summary
        assert summary['severity_counts']['info'] == 8
        assert summary['severity_counts']['error'] == 2
    
    def test_get_system_summary(self):
        """Test system summary generation."""
        # Log events for multiple merchants
        for merchant_id in ["M0005", "M0006", "M0007"]:
            for i in range(5):
                self.audit_trail.log_event(
                    merchant_id=merchant_id,
                    event_type="system_event",
                    details={"test": True}
                )
        
        summary = self.audit_trail.get_system_summary()
        
        assert summary['total_events'] == 15
        assert summary['unique_merchants'] == 3
        assert summary['average_events_per_merchant'] == 5.0
    
    def test_export_logs(self):
        """Test log export."""
        # Log some events
        for i in range(3):
            self.audit_trail.log_event(
                merchant_id="M0008",
                event_type="export_test",
                details={"export": True}
            )
        
        export_data = self.audit_trail.export_logs(format="json")
        
        assert export_data is not None
        assert isinstance(export_data, str)
        
        # Verify it's valid JSON
        import json
        parsed_data = json.loads(export_data)
        assert isinstance(parsed_data, list)
        assert len(parsed_data) >= 3

class TestRazorpayClient:
    """Test suite for Razorpay client (mocked)."""
    
    def setup_method(self):
        """Setup test fixtures."""
        from services.razorpay_client import RazorpayClient
        self.client = RazorpayClient()
    
    def test_simulate_failure(self):
        """Test failure simulation."""
        failure = self.client.simulate_failure("network")
        
        assert failure['success'] == False
        assert 'error' in failure
        assert failure['error_type'] == 'network'
        assert failure['simulated'] == True
    
    def test_random_failure_simulation(self):
        """Test random failure simulation."""
        failure = self.client.simulate_failure("random")
        
        assert failure['success'] == False
        assert failure['simulated'] == True
    
    def test_execute_with_retry_success(self):
        """Test retry logic with success."""
        def successful_func():
            return {"success": True, "data": "test"}
        
        result = self.client.execute_with_retry(successful_func)
        
        assert result['success'] == True
        assert result['data'] == 'test'
    
    def test_execute_with_retry_failure(self):
        """Test retry logic with failure."""
        def failing_func():
            return {"success": False, "error": "test error"}
        
        result = self.client.execute_with_retry(failing_func)
        
        assert result['success'] == False
        assert 'error' in result
        assert 'attempts' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])