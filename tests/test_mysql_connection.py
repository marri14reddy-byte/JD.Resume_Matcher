# Test to validate MySQL connection parameters are properly configured
# This test verifies the connection argument setup without requiring an actual MySQL server
# 
# NOTE: This test cannot currently run due to a pre-existing IndentationError in 
# resume_matcher_rag.py at line 3119. The test is kept here to document the expected
# behavior and can be run once the indentation error is fixed.
#
# Run with: python -m pytest -q tests/test_mysql_connection.py

import types
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Import target module
mod_path = Path(__file__).resolve().parent.parent / 'resume_matcher_rag.py'
spec = importlib.util.spec_from_file_location('resume_matcher_rag', str(mod_path))
mod = importlib.util.module_from_spec(spec)


def test_sync_from_mysql_connection_args():
    """Test that _sync_from_mysql sets proper connection arguments including timeouts"""
    
    # Mock mysql.connector to capture connection arguments
    mock_connector = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchmany.return_value = []
    mock_conn.cursor.return_value = mock_cursor
    mock_connector.connect.return_value = mock_conn
    
    # Mock streamlit
    mock_st = MagicMock()
    
    with patch.dict('sys.modules', {
        'mysql.connector': mock_connector,
        'mysql.connector.errors': MagicMock(),
        'streamlit': mock_st,
        'requests': MagicMock(),
    }):
        # Load the module with mocked dependencies
        spec.loader.exec_module(mod)
        
        # Call the function
        result = mod._sync_from_mysql(
            host='localhost',
            port=3306,
            user='test_user',
            password='test_pass',
            database='test_db',
            sql='SELECT * FROM resumes',
            limit=10,
            connect_timeout=10
        )
        
        # Verify connect was called
        assert mock_connector.connect.called, "MySQL connect should be called"
        
        # Get the connection arguments
        call_args = mock_connector.connect.call_args
        conn_args = call_args[1] if call_args[1] else call_args[0][0] if call_args[0] else {}
        
        # Verify critical timeout parameters are set
        assert 'read_timeout' in conn_args, "read_timeout should be set"
        assert 'write_timeout' in conn_args, "write_timeout should be set"
        assert conn_args['read_timeout'] == 300, "read_timeout should be 300 seconds"
        assert conn_args['write_timeout'] == 300, "write_timeout should be 300 seconds"
        
        # Verify connection stability parameters
        assert 'autocommit' in conn_args, "autocommit should be set"
        assert conn_args['autocommit'] is True, "autocommit should be True"
        assert 'use_pure' in conn_args, "use_pure should be set"
        assert conn_args['use_pure'] is True, "use_pure should be True"
        
        # Verify connection timeout is set
        assert 'connection_timeout' in conn_args, "connection_timeout should be set"
        assert conn_args['connection_timeout'] in [10, 30], "connection_timeout should be 10 or 30 seconds"


def test_sync_from_mysql_retry_on_operational_error():
    """Test that _sync_from_mysql retries on OperationalError"""
    
    # Create mock operational error
    mock_operational_error = type('OperationalError', (Exception,), {})
    
    # Mock mysql.connector
    mock_connector = MagicMock()
    mock_connector.errors.OperationalError = mock_operational_error
    
    # First two calls fail, third succeeds
    call_count = [0]
    def connect_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise mock_operational_error("Connection lost")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchmany.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn
    
    mock_connector.connect.side_effect = connect_side_effect
    
    # Mock streamlit and time
    mock_st = MagicMock()
    mock_time = MagicMock()
    
    with patch.dict('sys.modules', {
        'mysql.connector': mock_connector,
        'mysql.connector.errors': MagicMock(OperationalError=mock_operational_error),
        'streamlit': mock_st,
        'requests': MagicMock(),
        'time': mock_time,
    }):
        # Load the module with mocked dependencies
        spec.loader.exec_module(mod)
        
        # Call the function - should succeed after retries
        result = mod._sync_from_mysql(
            host='localhost',
            port=3306,
            user='test_user',
            password='test_pass',
            database='test_db',
            sql='SELECT * FROM resumes',
            limit=10,
            connect_timeout=10
        )
        
        # Verify retry happened
        assert call_count[0] == 3, "Should retry 3 times before success"
        assert mock_time.sleep.called, "Should sleep between retries"


if __name__ == '__main__':
    test_sync_from_mysql_connection_args()
    print("✓ Connection arguments test passed")
    test_sync_from_mysql_retry_on_operational_error()
    print("✓ Retry logic test passed")
    print("\nAll tests passed!")
