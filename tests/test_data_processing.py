import pandas as pd
import pytest
from src.model_trainer import (
    load_processed_data,
    split_data,
    train_and_tune_models,
)
import mlflow


# Mock data for testing
@pytest.fixture
def mock_data():
    data = pd.DataFrame(
        {
            "feature1": list(range(50)),
            "feature2": list(range(50, 100)),
            "is_high_risk": [0, 1] * 25,  # Alternating 0s and 1s for balanced
            # classes
        }
    )
    return data


def test_load_processed_data_existing_file(mock_data, tmp_path):
    """Test loading processed data from an existing file."""
    file_path = tmp_path / "test_data.csv"
    mock_data.to_csv(file_path, index=False)
    result = load_processed_data(str(tmp_path))
    assert result.shape == mock_data.shape
    pd.testing.assert_frame_equal(result, mock_data)


def test_load_processed_data_nonexistent_file():
    """
    Test loading processed data raises FileNotFoundError for nonexistent file.
    """
    with pytest.raises(FileNotFoundError):
        load_processed_data("nonexistent/path")


def test_split_data_correct_split(mock_data):
    """Test data splitting produces correct shapes and types."""
    X_train, X_test, y_train, y_test = split_data(mock_data)
    expected_train_size = len(mock_data) * 0.8
    assert (
        abs(len(X_train) - expected_train_size) <= 1
    )  # Allow for rounding tolerance
    assert len(X_test) == len(mock_data) - len(X_train)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)


def test_train_and_tune_models_logs_metrics(mock_data, mocker):
    """Test train_and_tune_models logs metrics to MLflow."""
    # Mock MLflow context and registration to avoid real MLflow calls
    mocker.patch.object(mlflow, "start_run")
    mock_log_metrics = mocker.patch.object(mlflow, "log_metrics")
    mock_log_params = mocker.patch.object(mlflow, "log_params")
    mock_log_model = mocker.patch.object(mlflow.sklearn, "log_model")
    mocker.patch.object(
        mlflow, "register_model", return_value=mocker.Mock(version="1")
    )  # Mock registration
    mock_set_experiment = mocker.patch.object(mlflow, "set_experiment")

    X_train, X_test, y_train, y_test = split_data(mock_data)
    train_and_tune_models(X_train, X_test, y_train, y_test, is_test=True)

    # Verify MLflow metrics, params, and model were logged
    assert mock_log_metrics.call_count > 0
    assert mock_log_params.call_count > 0
    assert mock_log_model.call_count > 0
    mock_set_experiment.assert_called_once_with("Credit_Risk_Modeling")
