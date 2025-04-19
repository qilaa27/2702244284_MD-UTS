
import pandas as pd
import pickle
import gzip
from sklearn.preprocessing import StandardScaler, RobustScaler

class BookingPredictor:
    def __init__(self, model_path, standard_scaler_path, robust_scaler_path, columns_path):
        if model_path:
            with gzip.open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        if standard_scaler_path:
            with open(standard_scaler_path, 'rb') as f:
                self.standard_scaler = pickle.load(f)
        if robust_scaler_path:
            with open(robust_scaler_path, 'rb') as f:
                self.robust_scaler = pickle.load(f)
        if columns_path:
            with open(columns_path, 'rb') as f:
                self.columns = pickle.load(f)

        self.cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.standard_cols = ['arrival_month', 'arrival_date']
        self.robust_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                            'no_of_week_nights', 'required_car_parking_space', 'lead_time',
                            'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
                            'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                            'no_of_special_requests']

    def preprocess(self, df):
        df = df.drop(columns=['Booking_ID'], errors='ignore')
        df = pd.get_dummies(df, columns=self.cat_cols, drop_first=True)
        df = df.reindex(columns=self.columns, fill_value=0)
        df[self.standard_cols] = self.standard_scaler.transform(df[self.standard_cols])
        df[self.robust_cols] = self.robust_scaler.transform(df[self.robust_cols])
        return df

    def predict(self, raw_df):
        X = self.preprocess(raw_df)
        prediction = self.model.predict(X)
        return prediction
