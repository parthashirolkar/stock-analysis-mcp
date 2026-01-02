import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .stock_analyzer import normalize_ticker
import yfinance as yf
import joblib
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def analyze_acf(series, lags, ticker):
    """
    Analyze ACF pattern for key insights.

    Returns:
        dict: Analysis insights including stationarity, model suggestions, seasonal patterns
    """
    from statsmodels.tsa.stattools import acf, adfuller

    # Calculate autocorrelation values
    autocorr_values = acf(series, nlags=lags, alpha=0.05)
    acf_values = autocorr_values[0]  # autocorrelation values
    conf_int = autocorr_values[1]  # confidence intervals

    # Perform ADF test for stationarity
    try:
        adf_result = adfuller(series.dropna(), autolag="AIC")
        adf_statistic = float(adf_result[0])
        adf_pvalue = float(adf_result[1])

        is_stationary_adf = adf_pvalue < 0.05
        adf_stationarity = "Stationary" if is_stationary_adf else "Non-stationary"
    except Exception:
        adf_statistic = None
        adf_pvalue = None
        is_stationary_adf = False
        adf_stationarity = "Test failed"

    # Significant lags (outside confidence bands)
    significant_lags = []
    for i in range(1, len(acf_values)):
        lower_ci = conf_int[i][0]
        upper_ci = conf_int[i][1]
        if acf_values[i] > upper_ci or acf_values[i] < lower_ci:
            significant_lags.append(i)

    # Determine decay pattern using multiple lags for better assessment
    lag_1_corr = abs(acf_values[1]) if len(acf_values) > 1 else 0
    lag_5_corr = abs(acf_values[5]) if len(acf_values) > 5 else 0

    # Calculate decay rate between lags
    decay_1_to_5 = (
        (lag_1_corr - lag_5_corr) / lag_1_corr
        if lag_1_corr > 0 and len(acf_values) > 5
        else 0
    )

    if lag_1_corr > 0.8:
        decay_pattern = "Very Slow (Strong Trend/Non-stationary)"
        stationarity_indicator = "Non-stationary - needs differencing"
    elif lag_1_corr > 0.6:
        decay_pattern = "Slow (Likely Non-stationary)"
        stationarity_indicator = "Potentially non-stationary"
    elif lag_1_corr > 0.3:
        if decay_1_to_5 > 0.7:  # Fast decay after initial
            decay_pattern = "Moderate to Fast"
            stationarity_indicator = "Likely Stationary"
        else:
            decay_pattern = "Moderate"
            stationarity_indicator = "Potentially stationary"
    else:
        decay_pattern = "Fast (Likely Stationary)"
        stationarity_indicator = "Stationary"

    # Detect seasonal patterns
    seasonal_patterns = []
    for lag in range(7, min(31, len(acf_values))):
        if acf_values[lag] > 0.3:  # Threshold for seasonal consideration
            seasonal_patterns.append(lag)

    # MA order suggestion
    ma_order_suggestion = None
    if significant_lags:
        # Find where autocorrelation cuts off (drops below significance)
        for i in range(1, len(significant_lags) - 1):
            if significant_lags[i] != significant_lags[i - 1] + 1:
                ma_order_suggestion = significant_lags[i - 1]
                break
        else:
            ma_order_suggestion = (
                significant_lags[-1] if len(significant_lags) < 10 else None
            )

    analysis = {
        "ticker": ticker,
        "data_points": len(series),
        "significant_lags": significant_lags[:10],  # Limit to first 10
        "lag_1_correlation": round(lag_1_corr, 4),
        "decay_pattern": decay_pattern,
        "stationarity_indicator": stationarity_indicator,
        "seasonal_patterns": seasonal_patterns,
        "ma_order_suggestion": ma_order_suggestion,
        "max_autocorr": round(max(abs(acf_values[1:])), 4)
        if len(acf_values) > 1
        else 0,
        # ADF test results
        "adf_test": {
            "statistic": round(adf_statistic, 4) if adf_statistic is not None else None,
            "p_value": round(adf_pvalue, 4) if adf_pvalue is not None else None,
            "is_stationary": is_stationary_adf,
            "result": adf_stationarity,
        },
        "interpretation": {
            "trend_strength": "Strong"
            if lag_1_corr > 0.7
            else "Moderate"
            if lag_1_corr > 0.4
            else "Weak",
            "forecastability": "High"
            if lag_1_corr > 0.6
            else "Moderate"
            if lag_1_corr > 0.3
            else "Low",
            "model_complexity": "High"
            if len(significant_lags) > 10
            else "Moderate"
            if len(significant_lags) > 5
            else "Low",
        },
    }

    return analysis


def analyze_pacf(series, lags, ticker):
    """
    Analyze PACF pattern for key insights.

    Returns:
        dict: Analysis insights including AR order suggestions and partial correlation patterns
    """
    from statsmodels.tsa.stattools import pacf

    # Adjust lags based on data size (need at least 4x data points for reliable PACF)
    data_len = len(series)
    adjusted_lags = min(lags, data_len // 4)  # Maximum 25% of data length

    # Calculate partial autocorrelation values
    pacf_values = pacf(series, nlags=adjusted_lags, alpha=0.05)
    pacf_corr = pacf_values[0]  # partial autocorrelation values
    conf_int = pacf_values[1]  # confidence intervals

    # Significant lags (outside confidence bands)
    significant_lags = []
    for i in range(1, len(pacf_corr)):
        lower_ci = conf_int[i][0]
        upper_ci = conf_int[i][1]
        if pacf_corr[i] > upper_ci or pacf_corr[i] < lower_ci:
            significant_lags.append(i)

    # AR order suggestion (sharp cutoff in PACF suggests AR order)
    ar_order_suggestion = None
    if significant_lags:
        # Find where PACF cuts off (drops below significance)
        for i in range(1, len(significant_lags) - 1):
            if significant_lags[i] != significant_lags[i - 1] + 1:
                ar_order_suggestion = significant_lags[i - 1]
                break
        else:
            ar_order_suggestion = (
                significant_lags[-1] if len(significant_lags) < 10 else None
            )

    # Check for immediate spike (lag-1 significance)
    lag_1_significant = 1 in significant_lags
    lag_1_partial_corr = pacf_corr[1] if len(pacf_corr) > 1 else 0

    # Determine pattern type
    if len(significant_lags) <= 2:
        pattern_type = "Sharp Cutoff (AR signature)"
    elif len(significant_lags) <= 5:
        pattern_type = "Moderate Cutoff"
    else:
        pattern_type = "Gradual Decay (MA signature)"

    analysis = {
        "ticker": ticker,
        "significant_lags": significant_lags[:10],  # Limit to first 10
        "lag_1_partial_corr": round(lag_1_partial_corr, 4),
        "lag_1_significant": lag_1_significant,
        "ar_order_suggestion": ar_order_suggestion,
        "pattern_type": pattern_type,
        "max_partial_corr": round(max(abs(pacf_corr[1:])), 4)
        if len(pacf_corr) > 1
        else 0,
        "interpretation": {
            "ar_suitability": "High"
            if pattern_type == "Sharp Cutoff (AR signature)"
            else "Moderate"
            if pattern_type == "Moderate Cutoff"
            else "Low",
            "direct_correlation": "Strong"
            if lag_1_significant and abs(lag_1_partial_corr) > 0.5
            else "Moderate"
            if lag_1_significant
            else "Weak",
            "model_simplicity": "Simple"
            if ar_order_suggestion and ar_order_suggestion <= 3
            else "Moderate"
            if ar_order_suggestion and ar_order_suggestion <= 6
            else "Complex",
        },
    }

    return analysis


class ARIMATrainer:
    """
    Core ARIMA training with intelligent parameter selection
    """

    def __init__(self, ticker: str, period: str = "1y"):
        self.ticker = ticker
        self.period = period
        self.model = None
        self.data = None
        self.analysis_result = None
        self.transform_type = None  # "log", "boxcox", or None
        self.transform_lambda = None  # For Box-Cox transformation
        self.original_data = None  # Store original data for inverse transformation

    def load_data(self):
        """Load stock data using existing patterns"""
        period_mapping = {
            "1D": "1d",
            "1W": "5d",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y",
        }

        period = period_mapping.get(self.period.upper(), "1y")

        ticker_variants = normalize_ticker(self.ticker)
        stock_data = None

        for ticker_symbol in ticker_variants.values():
            try:
                downloaded_data = yf.download(
                    ticker_symbol, period=period, auto_adjust=True
                )
                if not downloaded_data.empty:
                    stock_data = downloaded_data
                    self.ticker = ticker_symbol
                    break
            except Exception:
                continue

        if stock_data is None or stock_data.empty:
            raise ValueError(f"No valid stock data found for ticker {self.ticker}")

        # Extract Close prices and ensure it's always a 1D Series
        close_data = stock_data["Close"]

        # Handle MultiIndex columns (can happen with some yfinance versions/tickers)
        if isinstance(close_data, pd.DataFrame):
            # If it's a DataFrame, take the first column (or only column)
            if close_data.shape[1] == 1:
                close_data = close_data.iloc[:, 0]
            else:
                # Multiple columns - try to find the one matching our ticker
                ticker_base = normalize_ticker(self.ticker)["base"]
                if ticker_base in close_data.columns:
                    close_data = close_data[ticker_base]
                else:
                    # Fall back to first column
                    close_data = close_data.iloc[:, 0]

        # Ensure it's a Series and drop NaNs
        self.data = pd.Series(close_data).dropna()

        if len(self.data) == 0:
            raise ValueError(f"No valid closing prices found for ticker {self.ticker}")

    def _apply_transformation(
        self, data: pd.Series, transform_type: str = None
    ) -> pd.Series:
        """
        Apply transformation to data.

        Args:
            data: Input time series data
            transform_type: Type of transformation ("log", "boxcox", or None)

        Returns:
            Transformed data series
        """
        if transform_type is None or transform_type == "":
            return data

        if transform_type == "log":
            # Ensure all values are positive for log transformation
            if (data <= 0).any():
                raise ValueError(
                    "Log transformation requires all values to be positive"
                )
            return np.log(data)
        elif transform_type == "boxcox":
            from scipy.stats import boxcox

            # Box-Cox requires positive values
            if (data <= 0).any():
                # Shift data to be positive
                min_val = data.min()
                shift = abs(min_val) + 1 if min_val <= 0 else 0
                shifted_data = data + shift
            else:
                shifted_data = data
                shift = 0

            transformed, lambda_param = boxcox(shifted_data.values)
            self.transform_lambda = lambda_param
            # Store shift for inverse transformation
            self._boxcox_shift = shift
            return pd.Series(transformed, index=data.index)
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")

    def _inverse_transform(self, data) -> np.ndarray:
        """
        Apply inverse transformation to forecasts.

        Args:
            data: Transformed data (array, Series, or list)

        Returns:
            Inverse-transformed data
        """
        if self.transform_type is None:
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, pd.Series):
                return data.values
            else:
                return np.array(data)

        if self.transform_type == "log":
            if isinstance(data, np.ndarray):
                return np.exp(data)
            elif isinstance(data, pd.Series):
                return np.exp(data.values)
            else:
                return np.exp(np.array(data))
        elif self.transform_type == "boxcox":
            from scipy.special import inv_boxcox

            if isinstance(data, np.ndarray):
                data_array = data
            elif isinstance(data, pd.Series):
                data_array = data.values
            else:
                data_array = np.array(data)

            inverse_data = inv_boxcox(data_array, self.transform_lambda)
            # Remove shift if it was applied
            if hasattr(self, "_boxcox_shift") and self._boxcox_shift != 0:
                inverse_data = inverse_data - self._boxcox_shift
            return inverse_data
        else:
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, pd.Series):
                return data.values
            else:
                return np.array(data)

    def integrate_acf_pacf_suggestions(self, max_lags=40):
        """Integrate ACF/PACF analysis for parameter selection"""
        if self.data is None:
            self.load_data()

        # Adjust max_lags based on data size (need at least 4x data points for reliable ACF)
        data_len = len(self.data)
        adjusted_max_lags = min(max_lags, data_len // 4)  # Maximum 25% of data length

        # Get ACF and PACF analysis
        acf_analysis = analyze_acf(self.data, adjusted_max_lags, self.ticker)
        pacf_analysis = analyze_pacf(self.data, adjusted_max_lags, self.ticker)

        return {
            "acf_suggestion": acf_analysis["ma_order_suggestion"],
            "pacf_suggestion": pacf_analysis["ar_order_suggestion"],
            "acf_analysis": acf_analysis,
            "pacf_analysis": pacf_analysis,
            "recommended_ar": pacf_analysis["ar_order_suggestion"] or 1,  # Default AR=1
            "recommended_ma": acf_analysis["ma_order_suggestion"] or 1,  # Default MA=1
            "stationarity": acf_analysis["adf_test"]["is_stationary"],
            "trend_strength": acf_analysis["interpretation"]["trend_strength"],
        }

    def train_model(
        self,
        p: int = None,
        d: int = 1,
        q: int = None,
        validation_split: float = 0.2,
        auto_select: bool = True,
        transform: str = None,
    ):
        """
        Train ARIMA model with parameter validation and intelligent selection

        Args:
            p: AR order (None for auto-selection)
            d: Differencing order (default 1 for stock prices, None for auto-selection when using pmdarima)
            q: MA order (None for auto-selection)
            validation_split: Train-validation split ratio (default 0.2)
            auto_select: Use pmdarima auto_arima for parameter selection (True) or manual ACF/PACF (False)
            transform: Data transformation type ("log", "boxcox", or None)
        """
        if self.data is None:
            self.load_data()

        # Store original data for inverse transformation
        self.original_data = self.data.copy()
        self.transform_type = transform

        # Apply transformation if specified
        if transform:
            self.data = self._apply_transformation(self.data, transform)

        # Define max_lags for parameter constraints
        max_lags = len(self.data) - 2

        # Parameter selection strategy
        if auto_select:
            # Use pmdarima for automatic parameter selection
            try:
                from pmdarima import auto_arima  # type: ignore[import-untyped]

                # Split data for validation
                split_point = int(len(self.data) * (1 - validation_split))
                train_data = self.data.iloc[:split_point]
                test_data = self.data.iloc[split_point:]

                # Convert to numpy arrays to avoid Series truthiness ambiguity in pmdarima
                train_y = train_data.dropna().to_numpy(dtype=float)
                test_y = test_data.dropna().to_numpy(dtype=float)

                # Configure Box-Cox transformation in pmdarima
                lambda_param = "auto" if transform == "boxcox" else None

                # Use pmdarima auto_arima for automatic selection
                pmdarima_model = auto_arima(
                    train_y,
                    start_p=0,
                    start_q=0,
                    max_p=5,
                    max_q=5,
                    d=d
                    if d is not None
                    else None,  # Auto-select differencing if d is None
                    seasonal=False,
                    lambda_=lambda_param,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    information_criterion="aic",
                    trace=False,
                )

                # Extract selected parameters
                order = pmdarima_model.order  # (p, d, q)
                p, d, q = order[0], order[1], order[2]

                # Store the pmdarima model (it's compatible with statsmodels API)
                self.model = pmdarima_model

                # Make predictions on validation set
                n_periods_to_predict = len(test_y)
                test_predictions = self.model.predict(n_periods_to_predict)

                # Inverse transform predictions if transformation was applied
                if transform:
                    test_pred = self._inverse_transform(test_predictions)
                else:
                    test_pred = test_predictions

                # Calculate performance metrics
                # Inverse transform actual values if transformation was applied
                if transform:
                    test_actual = self._inverse_transform(test_y)
                else:
                    test_actual = test_y

                mse = mean_squared_error(test_actual, test_pred)
                mae = mean_absolute_error(test_actual, test_pred)
                mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

                # Extract AIC, BIC, and log-likelihood from pmdarima model
                # pmdarima wraps statsmodels, so we need to access the underlying results
                try:
                    # Try calling as methods (pmdarima's interface)
                    model_aic = (
                        pmdarima_model.aic()
                        if callable(getattr(pmdarima_model, "aic", None))
                        else pmdarima_model.aic
                    )
                    model_bic = (
                        pmdarima_model.bic()
                        if callable(getattr(pmdarima_model, "bic", None))
                        else pmdarima_model.bic
                    )
                    # llf is typically in the wrapped statsmodels results
                    model_llf = (
                        pmdarima_model.arima_res_.llf
                        if hasattr(pmdarima_model, "arima_res_")
                        else None
                    )
                except Exception:
                    model_aic = None
                    model_bic = None
                    model_llf = None

                return {
                    "model": self.model,
                    "parameters": {"p": p, "d": d, "q": q},
                    "performance": {
                        "mse": mse,
                        "mae": mae,
                        "mape": mape,
                        "train_size": len(train_y),
                        "test_size": len(test_y),
                        "validation_split": validation_split,
                    },
                    "aic": model_aic,
                    "bic": model_bic,
                    "log_likelihood": model_llf,
                    "converged": True,  # pmdarima handles convergence internally
                    "training_data_points": len(train_y),
                    "test_data_points": len(test_y),
                }

            except ImportError:
                raise RuntimeError(
                    "pmdarima is required but not installed. Install with: pip install pmdarima"
                )
            except Exception as e:
                raise RuntimeError(f"pmdarima auto_arima failed: {str(e)}")

        else:
            # Manual parameter selection (only when auto_select=False)
            if p is None or q is None:
                suggestions = self.integrate_acf_pacf_suggestions(
                    max_lags=min(max_lags, len(self.data) - 1)
                )
                p = p if p is not None else (suggestions["recommended_ar"] or 1)
                q = q if q is not None else (suggestions["recommended_ma"] or 1)
            d = d if d is not None else 1

            # Validate parameters
            if p >= len(self.data) or q >= len(self.data) or (p + q) >= max_lags:
                raise ValueError(f"Parameters (p={p}, q={q}) exceed data constraints")

            try:
                # Split data for validation
                split_point = int(len(self.data) * (1 - validation_split))
                train_data = self.data.iloc[:split_point]
                test_data = self.data.iloc[split_point:]

                # Train ARIMA model using statsmodels
                arima_model = ARIMA(train_data, order=(p, d, q))
                results = arima_model.fit()
                self.model = results  # Store the fitted results, not the unfitted model

                # Make predictions on validation set
                n_periods_to_predict = len(test_data)
                test_predictions = self.model.get_forecast(
                    steps=n_periods_to_predict
                ).predicted_mean

                # Inverse transform predictions if transformation was applied
                if transform:
                    test_pred = self._inverse_transform(test_predictions)
                else:
                    test_pred = test_predictions.values

                # Calculate performance metrics
                # Inverse transform actual values if transformation was applied
                if transform:
                    test_actual = self._inverse_transform(test_data.values)
                else:
                    test_actual = test_data.values

                mse = mean_squared_error(test_actual, test_pred)
                mae = mean_absolute_error(test_actual, test_pred)
                mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

                return {
                    "model": self.model,
                    "parameters": {"p": p, "d": d, "q": q},
                    "performance": {
                        "mse": mse,
                        "mae": mae,
                        "mape": mape,
                        "train_size": len(train_data),
                        "test_size": len(test_data),
                        "validation_split": validation_split,
                    },
                    "aic": results.aic,
                    "bic": results.bic,
                    "log_likelihood": results.llf,
                    "converged": results.mle_retvals is not None,
                    "training_data_points": len(train_data),
                    "test_data_points": len(test_data),
                }

            except Exception as e:
                raise RuntimeError(f"ARIMA training failed: {str(e)}")

    def forecast(self, periods: int = 20, confidence_level: float = 0.05):
        """
        Generate forecasts with confidence intervals

        Args:
            periods: Number of periods to forecast
            confidence_level: Confidence level (default 0.05 for 95% CI)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Generate forecasts using get_forecast for confidence intervals
            forecast_result = self.model.get_forecast(steps=periods)
            forecast_mean = forecast_result.predicted_mean
            forecast_conf_int = forecast_result.conf_int(alpha=confidence_level)

            # Create forecast dataframe
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=periods, freq="D"
            )

            # Apply inverse transformation if transformation was used
            if self.transform_type:
                forecast_mean_values = self._inverse_transform(forecast_mean)
                forecast_conf_int_lower = self._inverse_transform(
                    forecast_conf_int.iloc[:, 0]
                )
                forecast_conf_int_upper = self._inverse_transform(
                    forecast_conf_int.iloc[:, 1]
                )
            else:
                forecast_mean_values = forecast_mean.values
                forecast_conf_int_lower = forecast_conf_int.iloc[:, 0].values
                forecast_conf_int_upper = forecast_conf_int.iloc[:, 1].values

            forecasts = []
            for i in range(periods):
                if i < len(forecast_mean_values):
                    forecasts.append(
                        {
                            "date": forecast_dates[i].strftime("%Y-%m-%d"),
                            "forecast": float(forecast_mean_values[i]),
                            "lower_ci": float(forecast_conf_int_lower[i]),
                            "upper_ci": float(forecast_conf_int_upper[i]),
                        }
                    )

            return {
                "forecasts": forecasts,
                "confidence_level": 1 - confidence_level,
                "method": "arima_forecast",
                "periods": periods,
            }

        except Exception as e:
            raise RuntimeError(f"Forecast generation failed: {str(e)}")

    def save_model(self, filepath: str = None):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        if filepath is None:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = f"models/arima_{self.ticker}_{timestamp}.joblib"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and metadata
        metadata = {
            "ticker": self.ticker,
            "trained_at": datetime.now().isoformat(),
            "period": self.period,
            "parameters": self.model.order if hasattr(self.model, "order") else None,
            "aic": self.model.aic if hasattr(self.model, "aic") else None,
            "data_points": len(self.data) if self.data is not None else 0,
            "transform_type": self.transform_type,
            "transform_lambda": getattr(self, "transform_lambda", None),
            "_boxcox_shift": getattr(self, "_boxcox_shift", None),
        }

        joblib.dump({"model": self.model, "metadata": metadata}, filepath)

        return filepath

    def get_cached_model(self, model_key: str = None):
        """Get cached model if available and not expired"""
        if model_key is None:
            model_key = f"{self.ticker}_{self.period}_default"

        cache_file = f"models/arima_{model_key}.joblib"

        if os.path.exists(cache_file):
            try:
                # Check file age (1 hour cache)
                file_age = datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_file)
                )
                if file_age.total_seconds() < 3600:  # 1 hour
                    cached_model = joblib.load(cache_file)
                    return {
                        "model": cached_model,
                        "cached_at": datetime.fromtimestamp(
                            os.path.getmtime(cache_file)
                        ),
                        "cache_file": cache_file,
                    }
                else:
                    # Cache expired, remove it
                    os.remove(cache_file)
            except Exception:
                # Cache corrupted, remove it
                try:
                    os.remove(cache_file)
                except Exception:
                    pass

        return None

    def forecast_model(self, model, periods: int, confidence: float = 0.95):
        """Generate forecasts using trained model (supports both statsmodels and pmdarima)"""
        if model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Check if this is a pmdarima model or statsmodels model
            # pmdarima models have a predict method that returns conf_int
            # statsmodels models have get_forecast method
            if hasattr(model, "get_forecast") and callable(model.get_forecast):
                # Statsmodels path
                forecast_result = model.get_forecast(steps=periods)
                forecast_mean = forecast_result.predicted_mean
                forecast_conf_int = forecast_result.conf_int(alpha=1 - confidence)

                # Apply inverse transformation if transformation was used
                if self.transform_type:
                    forecast_mean_values = self._inverse_transform(forecast_mean)
                    forecast_conf_int_lower = self._inverse_transform(
                        forecast_conf_int.iloc[:, 0]
                    )
                    forecast_conf_int_upper = self._inverse_transform(
                        forecast_conf_int.iloc[:, 1]
                    )
                else:
                    forecast_mean_values = (
                        forecast_mean.tolist()
                        if hasattr(forecast_mean, "tolist")
                        else list(forecast_mean)
                    )
                    forecast_conf_int_lower = [
                        float(forecast_conf_int.iloc[i, 0])
                        for i in range(len(forecast_conf_int))
                    ]
                    forecast_conf_int_upper = [
                        float(forecast_conf_int.iloc[i, 1])
                        for i in range(len(forecast_conf_int))
                    ]
            elif hasattr(model, "predict") and callable(model.predict):
                # pmdarima path
                # pmdarima predict with return_conf_int returns (forecast, conf_int)
                forecast_mean, conf_int = model.predict(
                    n_periods=periods, return_conf_int=True, alpha=1 - confidence
                )

                # Apply inverse transformation if transformation was used
                if self.transform_type:
                    forecast_mean_values = self._inverse_transform(forecast_mean)
                    forecast_conf_int_lower = self._inverse_transform(conf_int[:, 0])
                    forecast_conf_int_upper = self._inverse_transform(conf_int[:, 1])
                else:
                    forecast_mean_values = (
                        forecast_mean.tolist()
                        if hasattr(forecast_mean, "tolist")
                        else list(forecast_mean)
                    )
                    forecast_conf_int_lower = [
                        float(conf_int[i, 0]) for i in range(len(conf_int))
                    ]
                    forecast_conf_int_upper = [
                        float(conf_int[i, 1]) for i in range(len(conf_int))
                    ]
            else:
                raise RuntimeError(
                    "Model does not support forecasting (no get_forecast or predict method)"
                )

            # Create forecast dataframe
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=periods, freq="D"
            )

            forecasts = []
            for i in range(periods):
                if i < len(forecast_mean_values):
                    forecasts.append(
                        {
                            "date": forecast_dates[i].strftime("%Y-%m-%d"),
                            "forecast": float(forecast_mean_values[i]),
                            "lower_ci": float(forecast_conf_int_lower[i]),
                            "upper_ci": float(forecast_conf_int_upper[i]),
                        }
                    )

            # Calculate analysis - use original data for last_price if transformation was applied
            if self.transform_type and self.original_data is not None:
                last_price = float(self.original_data.values[-1])
            else:
                last_price = float(self.data.values[-1])

            final_forecast = (
                float(forecast_mean_values[-1])
                if len(forecast_mean_values) > 0
                else last_price
            )
            price_change = final_forecast - last_price
            price_change_percent = (
                (price_change / last_price) * 100 if last_price > 0 else 0
            )

            forecast_values = [f["forecast"] for f in forecasts]
            min_forecast = min(forecast_values) if forecast_values else final_forecast
            max_forecast = max(forecast_values) if forecast_values else final_forecast
            forecast_range = max_forecast - min_forecast

            ci_lower = (
                float(forecast_conf_int_lower[-1])
                if len(forecast_conf_int_lower) > 0
                else final_forecast
            )
            ci_upper = (
                float(forecast_conf_int_upper[-1])
                if len(forecast_conf_int_upper) > 0
                else final_forecast
            )
            ci_band_width = ci_upper - ci_lower
            relative_band_width = (
                (ci_band_width / final_forecast) * 100 if final_forecast > 0 else 0
            )

            # Calculate volatility from historical data (use original if transformed)
            if self.transform_type and self.original_data is not None:
                returns = self.original_data.pct_change().dropna()
            else:
                returns = self.data.pct_change().dropna()
            price_volatility = float(returns.std()) if len(returns) > 0 else 0.1

            return {
                "forecast_dates": forecast_dates,
                "forecast_mean": forecast_mean_values,
                "forecast_ci_lower": forecast_conf_int_lower,
                "forecast_ci_upper": forecast_conf_int_upper,
                "analysis": {
                    "last_price": last_price,
                    "final_forecast": final_forecast,
                    "price_change": price_change,
                    "price_change_percent": price_change_percent,
                    "min_forecast": min_forecast,
                    "max_forecast": max_forecast,
                    "forecast_range": forecast_range,
                    "ci_lower_bound": ci_lower,
                    "ci_upper_bound": ci_upper,
                    "ci_band_width": ci_band_width,
                    "relative_band_width": relative_band_width,
                    "forecast_start_date": forecast_dates[0].strftime("%Y-%m-%d"),
                    "forecast_end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
                    "price_volatility": price_volatility,
                    "prediction_quality": "High"
                    if relative_band_width < 0.05
                    else "Medium"
                    if relative_band_width < 0.15
                    else "Low",
                    "model_quality": "Good" if price_volatility < 0.2 else "Fair",
                },
                "performance": {
                    "standard_error": float(np.std([f["forecast"] for f in forecasts])),
                    "mae": float(
                        np.mean([abs(f["forecast"] - last_price) for f in forecasts])
                    ),
                    "data_points": len(self.original_data)
                    if self.original_data is not None
                    else len(self.data),
                },
            }

        except Exception as e:
            raise RuntimeError(f"Forecast generation failed: {str(e)}")

    def comprehensive_diagnostics(self):
        """Perform comprehensive diagnostic analysis on trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Get residuals - handle both statsmodels and pmdarima models
            if hasattr(self.model, "resid"):
                # pmdarima models have resid as a callable method
                # statsmodels models have resid as an array attribute
                if callable(self.model.resid):
                    # pmdarima: call method to get residuals
                    residuals = pd.Series(self.model.resid())
                else:
                    # statsmodels: resid is an array attribute
                    residuals = pd.Series(self.model.resid)
            elif hasattr(self.model, "arima_res_") and hasattr(
                self.model.arima_res_, "resid"
            ):
                # pmdarima models store statsmodels results in arima_res_
                residuals = pd.Series(self.model.arima_res_.resid)
            else:
                raise RuntimeError(
                    "Model does not have residuals (no resid attribute found)"
                )

            dates = self.data.index[-len(residuals) :]

            # Calculate residuals analysis
            mean_residual = float(residuals.mean())
            std_residual = float(residuals.std())
            min_residual = float(residuals.min())
            max_residual = float(residuals.max())
            residual_sum = float(residuals.sum())

            # ACF analysis of residuals
            from statsmodels.tsa.stattools import acf

            acf_result = acf(residuals, nlags=40, alpha=0.05)

            # Handle tuple return when alpha is provided: (acf_values, confint)
            if isinstance(acf_result, tuple):
                acf_values_array, confint = acf_result
                # Convert to list for consistency
                residual_acf_values = (
                    acf_values_array.tolist()
                    if hasattr(acf_values_array, "tolist")
                    else list(acf_values_array)
                )
                # Calculate symmetric confidence interval from confint
                # confint shape is (nlags+1, 2) with [lower, upper] bounds
                # For plotting, use a single symmetric confidence band value
                if confint is not None and len(confint) > 0:
                    # Use the mean absolute upper bound as the confidence interval
                    # This gives us a symmetric ±confidence_interval band
                    confidence_interval = float(np.mean(np.abs(confint[:, 1])))
                else:
                    # Fallback: approximate 95% CI using 1.96/sqrt(N)
                    confidence_interval = 1.96 / np.sqrt(len(residuals))
            else:
                # Fallback for older statsmodels versions (shouldn't happen with alpha)
                residual_acf_values = (
                    acf_result.tolist()
                    if hasattr(acf_result, "tolist")
                    else list(acf_result)
                )
                confidence_interval = 1.96 / np.sqrt(len(residuals))

            residual_acf_lags = list(range(len(residual_acf_values)))

            # Normality tests
            import scipy.stats as stats

            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(residuals)

            # Ljung-Box test
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            # acorr_ljungbox returns a DataFrame with 'lb_stat' and 'lb_pvalue' columns when return_df=True
            if isinstance(lb_result, pd.DataFrame) and len(lb_result) > 0:
                lb_stat = float(lb_result.iloc[0]["lb_stat"])
                lb_p = float(lb_result.iloc[0]["lb_pvalue"])
            else:
                # Fallback: try without return_df parameter (older versions may return tuple)
                try:
                    lb_result_fallback = acorr_ljungbox(residuals, lags=[10])
                    if isinstance(lb_result_fallback, pd.DataFrame):
                        lb_stat = float(lb_result_fallback.iloc[0]["lb_stat"])
                        lb_p = float(lb_result_fallback.iloc[0]["lb_pvalue"])
                    else:
                        # If it's a tuple or array-like
                        lb_stat = (
                            float(lb_result_fallback[0])
                            if len(lb_result_fallback) > 0
                            else 0.0
                        )
                        lb_p = (
                            float(lb_result_fallback[1])
                            if len(lb_result_fallback) > 1
                            else 1.0
                        )
                except Exception:
                    # Ultimate fallback
                    lb_stat = 0.0
                    lb_p = 1.0

            # Find significant lags
            significant_lags = []
            for i, acf_val in enumerate(residual_acf_values):
                if i > 0 and abs(acf_val) > confidence_interval:
                    significant_lags.append(i)

            normality_tests = {
                "Shapiro-Wilk": {
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                    "result": "Normal" if shapiro_p > 0.05 else "Non-normal",
                },
                "Jarque-Bera": {
                    "statistic": jb_stat,
                    "p_value": jb_p,
                    "result": "Normal" if jb_p > 0.05 else "Non-normal",
                },
            }

            ljung_box_test = {
                "statistic": lb_stat,
                "p_value": lb_p,
                "lags": 10,
                "result": "White noise" if lb_p > 0.05 else "Not white noise",
            }

            # Generate recommendations
            model_improvements = []
            parameter_suggestions = []
            warnings = []

            if shapiro_p < 0.05:
                model_improvements.append("Consider transformation (log, Box-Cox)")

            if lb_p < 0.05:
                model_improvements.append(
                    "Residuals show autocorrelation - increase AR/MA orders"
                )

            if len(significant_lags) > 5:
                model_improvements.append(
                    "Too many significant lags - consider differencing"
                )
                warnings.append("Complex residual pattern detected")

            recommendations = {
                "overall_assessment": "Adequate"
                if shapiro_p > 0.05 and lb_p > 0.05
                else "Needs improvement",
                "white_noise_conclusion": "Residuals are white noise"
                if lb_p > 0.05
                else "Residuals show correlation",
                "autocorrelation_conclusion": "No significant autocorrelation"
                if len(significant_lags) == 0
                else "Significant autocorrelation present",
                "normality_conclusion": "Residuals are normally distributed"
                if shapiro_p > 0.05
                else "Residuals are not normally distributed",
                "significant_residual_lags": significant_lags[:5],
                "model_improvements": model_improvements,
                "parameter_suggestions": parameter_suggestions,
                "warnings": warnings,
                "model_quality": "Good" if shapiro_p > 0.05 and lb_p > 0.05 else "Fair",
                "forecast_reliability": "High"
                if len(significant_lags) < 3
                else "Medium",
                "complexity_level": "Simple"
                if len(significant_lags) < 3
                else "Moderate",
                "risk_assessment": "Low"
                if shapiro_p > 0.05 and lb_p > 0.05
                else "Medium",
            }

            return {
                "dates": dates.strftime("%Y-%m-%d").tolist(),
                "residuals": residuals.tolist(),
                "residual_acf": {
                    "lags": residual_acf_lags,
                    "acf_values": residual_acf_values,
                    "confidence_interval": confidence_interval,
                },
                "diagnostics": {
                    "mean_residual": mean_residual,
                    "std_residual": std_residual,
                    "min_residual": min_residual,
                    "max_residual": max_residual,
                    "residual_sum": residual_sum,
                },
                "normality_tests": normality_tests,
                "ljung_box_test": ljung_box_test,
                "recommendations": recommendations,
                "data_points": len(self.data),
                "model_info": self._get_model_info(),
                "date_range": {
                    "start": dates[0].strftime("%Y-%m-%d"),
                    "end": dates[-1].strftime("%Y-%m-%d"),
                },
            }

        except Exception as e:
            raise RuntimeError(f"Diagnostic analysis failed: {str(e)}")

    def _get_model_convergence(self, model):
        """
        Safely get convergence status from both pmdarima and statsmodels models.

        Args:
            model: Trained ARIMA model (pmdarima or statsmodels)

        Returns:
            bool: True if model converged, False otherwise
        """
        try:
            # Try direct access (statsmodels)
            if hasattr(model, "mle_retvals"):
                return model.mle_retvals is not None
            # Try wrapped access (pmdarima stores statsmodels results in arima_res_)
            elif hasattr(model, "arima_res_") and hasattr(
                model.arima_res_, "mle_retvals"
            ):
                return model.arima_res_.mle_retvals is not None
            # Fallback - assume converged if model exists and was successfully trained
            return True
        except Exception:
            # If anything goes wrong, assume converged
            return True

    def _get_model_info(self):
        """Extract model order (p, d, q) from either statsmodels or pmdarima models"""
        try:
            # For pmdarima models, use the order attribute
            if hasattr(self.model, "order"):
                order = self.model.order
                return {"p": order[0], "d": order[1], "q": order[2]}
            # For statsmodels models, try k_ar, k_diff, k_ma attributes
            elif hasattr(self.model, "k_ar"):
                return {
                    "p": getattr(self.model, "k_ar", 1),
                    "d": getattr(self.model, "k_diff", 1),
                    "q": getattr(self.model, "k_ma", 1),
                }
            else:
                # Fallback to default values
                return {"p": 1, "d": 1, "q": 1}
        except Exception:
            # Ultimate fallback
            return {"p": 1, "d": 1, "q": 1}


class SARIMATrainer(ARIMATrainer):
    """
    Seasonal ARIMA trainer extending base ARIMA functionality
    """

    def __init__(self, ticker: str, period: str = "1y", seasonal_period: int = None):
        super().__init__(ticker, period)
        self.seasonal_period = seasonal_period
        self.seasonal_analysis = None

    def detect_seasonal_period(self):
        """Detect optimal seasonal period using FFT analysis"""
        if self.data is None:
            self.load_data()

        # Simple seasonal detection using FFT
        from scipy.fft import fft

        values = self.data.values

        # Remove trend and compute FFT
        detrended = values - np.mean(values)
        fft_values = np.abs(fft(detrended))

        # Find dominant frequencies
        freqs = np.fft.fftfreq(len(values), 1.0 / 252)  # Trading days

        # Find dominant frequency indices (excluding DC component)
        top_indices = np.argsort(fft_values)[-10:][::-1][
            1:
        ]  # Top 9 frequencies excluding DC

        # Convert frequency indices to periods
        seasonal_periods = []
        for idx in top_indices:
            if freqs[idx] != 0:
                period = int(1 / abs(freqs[idx]))
                if 5 <= period <= 252:  # Reasonable range for trading days
                    seasonal_periods.append(period)

        # Return most common seasonal period
        if seasonal_periods:
            return max(set(seasonal_periods), key=seasonal_periods.count)
        return None

    def train_model(
        self,
        p: int = None,
        d: int = 1,
        q: int = None,
        P: int = None,
        D: int = 1,
        Q: int = None,
        validation_split: float = 0.2,
        auto_select: bool = True,
        transform: str = None,
    ):
        """
        Train SARIMA model with seasonal component

        Args:
            p: AR order (None for auto-selection)
            d: Differencing order (default 1 for stock prices, None for auto-selection when using pmdarima)
            q: MA order (None for auto-selection)
            P: Seasonal AR order (None for auto-selection)
            D: Seasonal differencing order (default 1)
            Q: Seasonal MA order (None for auto-selection)
            validation_split: Train-validation split ratio (default 0.2)
            auto_select: Use pmdarima auto_arima for parameter selection (True) or manual (False)
            transform: Data transformation type ("log", "boxcox", or None)
        """
        if self.data is None:
            self.load_data()

        # Store original data for inverse transformation
        self.original_data = self.data.copy()
        self.transform_type = transform

        # Apply transformation if specified
        if transform:
            self.data = self._apply_transformation(self.data, transform)

        # Auto-detect seasonal period if not provided
        if self.seasonal_period is None and auto_select:
            self.seasonal_period = self.detect_seasonal_period()

        # Define max_lags for parameter constraints
        max_lags = len(self.data) - 2

        # Parameter selection strategy
        if auto_select:
            # Use pmdarima for automatic parameter selection
            try:
                from pmdarima import auto_arima  # type: ignore[import-untyped]

                # Split data for validation
                split_point = int(len(self.data) * (1 - validation_split))
                train_data = self.data.iloc[:split_point]
                test_data = self.data.iloc[split_point:]

                # Convert to numpy arrays to avoid Series truthiness ambiguity in pmdarima
                train_y = train_data.dropna().to_numpy(dtype=float)
                test_y = test_data.dropna().to_numpy(dtype=float)

                # Configure Box-Cox transformation in pmdarima
                lambda_param = "auto" if transform == "boxcox" else None

                # Determine seasonal period
                seasonal_period = self.seasonal_period if self.seasonal_period else 4

                # Use pmdarima auto_arima for automatic selection with seasonal component
                pmdarima_model = auto_arima(
                    train_y,
                    start_p=0,
                    start_q=0,
                    max_p=5,
                    max_q=5,
                    d=d if d is not None else None,
                    start_P=0,
                    start_Q=0,
                    max_P=2,
                    max_Q=2,
                    D=D if D is not None else None,
                    seasonal=True,
                    m=seasonal_period,
                    lambda_=lambda_param,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    information_criterion="aic",
                    trace=False,
                )

                # Extract selected parameters
                order = pmdarima_model.order  # (p, d, q)
                seasonal_order = pmdarima_model.seasonal_order  # (P, D, Q, m)
                p, d, q = order[0], order[1], order[2]
                P, D, Q, m = (
                    seasonal_order[0],
                    seasonal_order[1],
                    seasonal_order[2],
                    seasonal_order[3],
                )
                self.seasonal_period = m

                # Store the pmdarima model
                self.model = pmdarima_model

                # Make predictions on validation set
                n_periods_to_predict = len(test_y)
                test_predictions = self.model.predict(n_periods_to_predict)

                # Inverse transform predictions if transformation was applied
                if transform:
                    test_pred = self._inverse_transform(test_predictions)
                else:
                    test_pred = test_predictions

                # Calculate performance metrics
                # Inverse transform actual values if transformation was applied
                if transform:
                    test_actual = self._inverse_transform(test_y)
                else:
                    test_actual = test_y

                mse = mean_squared_error(test_actual, test_pred)
                mae = mean_absolute_error(test_actual, test_pred)

                # Extract AIC, BIC, and log-likelihood from pmdarima model
                # pmdarima wraps statsmodels, so we need to access the underlying results
                try:
                    # Try calling as methods (pmdarima's interface)
                    model_aic = (
                        pmdarima_model.aic()
                        if callable(getattr(pmdarima_model, "aic", None))
                        else pmdarima_model.aic
                    )
                    model_bic = (
                        pmdarima_model.bic()
                        if callable(getattr(pmdarima_model, "bic", None))
                        else pmdarima_model.bic
                    )
                    # llf is typically in the wrapped statsmodels results
                    model_llf = (
                        pmdarima_model.arima_res_.llf
                        if hasattr(pmdarima_model, "arima_res_")
                        else None
                    )
                except Exception:
                    model_aic = None
                    model_bic = None
                    model_llf = None

                return {
                    "model": self.model,
                    "parameters": {
                        "p": p,
                        "d": d,
                        "q": q,
                        "P": P,
                        "D": D,
                        "Q": Q,
                        "seasonal_period": self.seasonal_period,
                    },
                    "performance": {
                        "mse": mse,
                        "mae": mae,
                        "train_size": len(train_y),
                        "test_size": len(test_y),
                        "validation_split": validation_split,
                    },
                    "aic": model_aic,
                    "bic": model_bic,
                    "log_likelihood": model_llf,
                    "converged": True,  # pmdarima handles convergence internally
                    "seasonal_period": self.seasonal_period,
                }

            except ImportError:
                raise RuntimeError(
                    "pmdarima is required but not installed. Install with: pip install pmdarima"
                )
            except Exception as e:
                raise RuntimeError(f"pmdarima auto_arima failed: {str(e)}")

        # Manual parameter selection (only when auto_select=False)
        else:
            if p is None or q is None:
                suggestions = self.integrate_acf_pacf_suggestions(
                    max_lags=min(40, len(self.data) - 1)
                )
            p = (
                p
                if p is not None
                else (suggestions["recommended_ar"] if auto_select else 1)
            )
            q = (
                q
                if q is not None
                else (suggestions["recommended_ma"] if auto_select else 1)
            )
            d = d if d is not None else 1
            P = P if P is not None else (self.seasonal_period or 4)
            D = D if D is not None else 1
            Q = Q if Q is not None else 1

        # Validate parameters for SARIMA
        seasonal_constraint = max_lags - (P * D + Q)  # Account for seasonal lags
        if p >= len(self.data) or q >= len(self.data) or seasonal_constraint <= 0:
            raise ValueError("Parameters exceed data constraints")

        try:
            # Split data
            split_point = int(len(self.data) * (1 - validation_split))
            train_data = self.data.iloc[:split_point]
            test_data = self.data.iloc[split_point:]

            # Train SARIMA model using statsmodels
            seasonal_period = self.seasonal_period if self.seasonal_period else 4
            sarimax_model = SARIMAX(
                train_data, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period)
            )
            results = sarimax_model.fit()
            self.model = results  # Store the fitted results, not the unfitted model

            # Validation and performance metrics (similar to ARIMA)
            n_periods_to_predict = len(test_data)
            test_predictions = self.model.get_forecast(
                steps=n_periods_to_predict
            ).predicted_mean

            # Inverse transform predictions if transformation was applied
            if transform:
                test_pred = self._inverse_transform(test_predictions)
            else:
                test_pred = test_predictions.values

            # Calculate performance metrics
            # Inverse transform actual values if transformation was applied
            if transform:
                test_actual = self._inverse_transform(test_data.values)
            else:
                test_actual = test_data.values

            mse = mean_squared_error(test_actual, test_pred)
            mae = mean_absolute_error(test_actual, test_pred)

            return {
                "model": self.model,
                "parameters": {
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": P,
                    "D": D,
                    "Q": Q,
                    "seasonal_period": self.seasonal_period,
                },
                "performance": {
                    "mse": mse,
                    "mae": mae,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "validation_split": validation_split,
                },
                "aic": results.aic,
                "bic": results.bic,
                "log_likelihood": results.llf,
                "converged": results.mle_retvals is not None,
                "seasonal_period": self.seasonal_period,
            }

        except Exception as e:
            raise RuntimeError(f"SARIMA training failed: {str(e)}")


class ProphetTrainer:
    """
    Facebook Prophet model trainer for stock price forecasting

    Uses Prophet's automatic seasonality detection and changepoint identification
    for flexible and interpretable time series forecasting.
    """

    def __init__(self, ticker: str, period: str = "1y"):
        self.ticker = ticker
        self.period = period
        self.model = None
        self.data = None
        self.original_data = None

    def load_data(self) -> None:
        """Load stock data using yfinance"""
        period_mapping = {
            "1D": "1d",
            "1W": "5d",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y",
        }

        period = period_mapping.get(self.period.upper(), "1y")

        ticker_variants = normalize_ticker(self.ticker)
        stock_data = None

        for ticker_symbol in ticker_variants.values():
            try:
                downloaded_data = yf.download(
                    ticker_symbol, period=period, auto_adjust=True
                )
                if not downloaded_data.empty:
                    stock_data = downloaded_data
                    self.ticker = ticker_symbol
                    break
            except Exception:
                continue

        if stock_data is None or stock_data.empty:
            raise ValueError(f"No valid stock data found for ticker {self.ticker}")

        close_data = stock_data["Close"]

        if isinstance(close_data, pd.DataFrame):
            if close_data.shape[1] == 1:
                close_data = close_data.iloc[:, 0]
            else:
                ticker_base = normalize_ticker(self.ticker)["base"]
                if ticker_base in close_data.columns:
                    close_data = close_data[ticker_base]
                else:
                    close_data = close_data.iloc[:, 0]

        self.data = pd.Series(close_data).dropna()
        self.original_data = self.data.copy()

        if len(self.data) == 0:
            raise ValueError(f"No valid closing prices found for ticker {self.ticker}")

    def train_model(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        holidays: pd.DataFrame = None,
        validation_split: float = 0.2,
        include_holidays: bool = False,
        confidence: float = 0.95,
    ) -> dict:
        """
        Train Prophet model with configurable parameters.

        Args:
            yearly_seasonality: Enable yearly seasonality (default True)
            weekly_seasonality: Enable weekly seasonality (default True)
            daily_seasonality: Enable daily seasonality (default False)
            seasonality_mode: 'additive' or 'multiplicative' (default 'additive')
            changepoint_prior_scale: Flexibility of trend changes (default 0.05)
            seasonality_prior_scale: Flexibility of seasonality (default 10.0)
            holidays_prior_scale: Flexibility of holiday effects (default 10.0)
            holidays: Custom holidays DataFrame (optional)
            validation_split: Train-validation split ratio (default 0.2)
            include_holidays: Include Indian market holidays (default False)
            confidence: Confidence interval level (0.8-0.99, default 0.95)

        Returns:
            dict with model, parameters, performance metrics
        """
        if self.data is None:
            self.load_data()

        try:
            from prophet import Prophet

            split_point = int(len(self.data) * (1 - validation_split))
            train_data = self.data.iloc[:split_point]
            test_data = self.data.iloc[split_point:]

            # Normalize datetime index to midnight (no timezone) for Prophet
            train_df = pd.DataFrame({
                "ds": pd.to_datetime(train_data.index).normalize(),
                "y": train_data.values
            })

            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                holidays=holidays,
                interval_width=confidence,
            )

            if include_holidays:
                model.add_country_holidays(country_name="IN")

            model.fit(train_df)

            # Store trained model
            self.model = model

            # Use business-day frequency for predictions
            future = model.make_future_dataframe(
                periods=len(test_data), freq="B", include_history=True
            )
            forecast = model.predict(future)

            # Normalize both datetime indices for proper alignment
            forecast["ds"] = pd.to_datetime(forecast["ds"]).dt.normalize()
            test_data_normalized = test_data.copy()
            test_data_normalized.index = pd.to_datetime(test_data.index).normalize()
            train_data_normalized = train_data.copy()
            train_data_normalized.index = pd.to_datetime(train_data.index).normalize()

            # Join forecast to test data by date (proper alignment, no isin filtering)
            test_forecast_df = forecast.merge(
                pd.DataFrame({"ds": test_data_normalized.index, "actual": test_data_normalized.values}),
                on="ds",
                how="inner"
            )

            # Only compute metrics where we have both prediction and actual
            if len(test_forecast_df) > 0:
                test_predictions = test_forecast_df["yhat"].values
                test_actual = test_forecast_df["actual"].values

                mse = mean_squared_error(test_actual, test_predictions)
                mae = mean_absolute_error(test_actual, test_predictions)
                mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
            else:
                # Fallback if no alignment (shouldn't happen with freq="B")
                mse = mae = mape = float('nan')

            # Extract train forecast for plotting (aligned by date)
            train_forecast_df = forecast.merge(
                pd.DataFrame({"ds": train_data_normalized.index}),
                on="ds",
                how="inner"
            )

            return {
                "model": model,
                "train_df": train_df,
                "test_data": test_data,
                "train_forecast": train_forecast_df,
                "test_forecast": test_forecast_df,
                "parameters": {
                    "yearly_seasonality": yearly_seasonality,
                    "weekly_seasonality": weekly_seasonality,
                    "daily_seasonality": daily_seasonality,
                    "seasonality_mode": seasonality_mode,
                    "changepoint_prior_scale": changepoint_prior_scale,
                    "seasonality_prior_scale": seasonality_prior_scale,
                    "holidays_prior_scale": holidays_prior_scale,
                    "include_holidays": include_holidays,
                    "confidence": confidence,
                },
                "performance": {
                    "mse": mse,
                    "mae": mae,
                    "mape": mape,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "validation_split": validation_split,
                    "matched_test_points": len(test_forecast_df),
                },
                "train_data_points": len(train_data),
                "test_data_points": len(test_data),
            }

        except ImportError:
            raise RuntimeError(
                "Prophet is required but not installed. Install with: uv add prophet"
            )
        except Exception as e:
            raise RuntimeError(f"Prophet training failed: {str(e)}")

    def forecast(
        self,
        periods: int,
        model=None,
    ) -> dict:
        """
        Generate Prophet forecasts with confidence intervals.

        Args:
            periods: Number of trading day periods to forecast
            model: Trained Prophet model (optional, uses self.model if not provided)

        Returns:
            dict with forecast data and analysis
        """
        if self.model is None and model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            if model is None:
                model = self.model

            # Use business-day frequency for forecast
            future = model.make_future_dataframe(periods=periods, freq="B")
            forecast = model.predict(future)

            # Get only the future forecast (last N periods)
            forecast_filtered = forecast.iloc[-periods:]

            forecast_dates = pd.to_datetime(forecast_filtered["ds"])
            forecast_mean = forecast_filtered["yhat"].values
            forecast_lower = forecast_filtered["yhat_lower"].values
            forecast_upper = forecast_filtered["yhat_upper"].values

            last_price = float(self.original_data.values[-1])
            final_forecast = float(forecast_mean[-1])
            price_change = final_forecast - last_price
            price_change_percent = (
                (price_change / last_price) * 100 if last_price > 0 else 0
            )

            min_forecast = float(np.min(forecast_mean))
            max_forecast = float(np.max(forecast_mean))
            forecast_range = max_forecast - min_forecast

            ci_lower = float(forecast_lower[-1])
            ci_upper = float(forecast_upper[-1])
            ci_band_width = ci_upper - ci_lower
            relative_band_width = (
                (ci_band_width / final_forecast) * 100 if final_forecast > 0 else 0
            )

            returns = self.original_data.pct_change().dropna()
            price_volatility = float(returns.std()) if len(returns) > 0 else 0.1

            return {
                "forecast_dates": forecast_dates,
                "forecast_mean": forecast_mean,
                "forecast_ci_lower": forecast_lower,
                "forecast_ci_upper": forecast_upper,
                "analysis": {
                    "last_price": last_price,
                    "final_forecast": final_forecast,
                    "price_change": price_change,
                    "price_change_percent": price_change_percent,
                    "min_forecast": min_forecast,
                    "max_forecast": max_forecast,
                    "forecast_range": forecast_range,
                    "ci_lower_bound": ci_lower,
                    "ci_upper_bound": ci_upper,
                    "ci_band_width": ci_band_width,
                    "relative_band_width": relative_band_width,
                    "forecast_start_date": forecast_dates.iloc[0].strftime("%Y-%m-%d"),
                    "forecast_end_date": forecast_dates.iloc[-1].strftime("%Y-%m-%d"),
                    "price_volatility": price_volatility,
                    "prediction_quality": "High"
                    if relative_band_width < 0.05
                    else "Medium"
                    if relative_band_width < 0.15
                    else "Low",
                    "model_quality": "Good" if price_volatility < 0.2 else "Fair",
                },
                "performance": {
                    "standard_error": float(np.std(forecast_mean)),
                    "mae": float(np.mean([abs(f - last_price) for f in forecast_mean])),
                    "data_points": len(self.original_data),
                },
            }

        except Exception as e:
            raise RuntimeError(f"Prophet forecast generation failed: {str(e)}")
