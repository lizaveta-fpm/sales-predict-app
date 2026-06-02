import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler


class SalesLSTMWithEmbeddings(nn.Module):
    def __init__(self, num_stores, num_families, store_dim, family_dim, num_numerical, hidden_size=32):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, store_dim)
        self.family_emb = nn.Embedding(num_families, family_dim)
        
        # Считаем суммарный размер 
        self.lstm = nn.LSTM(store_dim + family_dim + num_numerical, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, store_ids, family_ids, numerical_features):
        # Извлекаем эмбеддинги и склеиваем всё в один тензор 
        x = torch.cat([self.store_emb(store_ids), self.family_emb(family_ids), numerical_features], dim=-1)
        
        #Выходная последовательность и финальные состояния памяти (hidden state, cell state)
        output_sequence, (hidden_state, cell_state) = self.lstm(x)
        # берем последний шаг из выходной последовательности:
        return self.fc(output_sequence[:, -1, :])


def load_data():
    
    df = pd.read_csv('train.csv', parse_dates=['date'])
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
    
    oil_df = pd.read_csv('oil.csv', parse_dates=['date'])
    store_df = pd.read_csv('stores.csv')
    holiday_df = pd.read_csv('holidays_events.csv', parse_dates=['date']).drop_duplicates(subset=['date'])
    
    # Календарные признаки
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Магазины
    df = df.merge(store_df[['store_nbr', 'cluster', 'type']], on='store_nbr', how='left')
    df['cluster'] = df['cluster'].fillna(0).astype(int)
    df['store_type'] = df['type'].fillna('Unknown').astype('category').cat.codes
    df = df.drop(columns=['type'])
    
    # Нефть 
    df = df.merge(oil_df, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill().fillna(0)
    
    # Праздники
    holiday_df['is_holiday'] = 1
    df = df.merge(holiday_df[['date', 'is_holiday']], on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    
    return df


def main():
    df_base = load_data()
    
    horizons = [7, 14, 16, 21, 30, 60, 90]
    models = ["XGBoost", "CatBoost", "LSTM"]
    
    master_storage = []
    importance_storage = []
    
    for horizon in horizons:
        print(f"\n Обработка горизонта: {horizon} дней ")
        df = df_base.copy()
        
        # Динамические лаги под текущий горизонт (внутри групп)
        grouped = df.groupby(['store_nbr', 'family'], observed=True)
        df['lag_base'] = grouped['sales'].shift(horizon)
        df['lag_base_1'] = grouped['sales'].shift(horizon + 1)
        df['lag_base_7'] = grouped['sales'].shift(horizon + 7)
        df['rolling_mean_7'] = grouped['sales'].transform(lambda x: x.shift(horizon).rolling(7).mean())
        
        df = df.dropna().reset_index(drop=True)
        
        # Индексы для эмбеддингов LSTM
        df['store_code'] = df['store_nbr'].astype('category').cat.codes
        df['family_code'] = df['family'].astype('category').cat.codes
        num_stores = df['store_code'].nunique()
        num_families = df['family_code'].nunique()
        
        # сплит выборки
        last_date = df['date'].max()
        split_date = last_date - pd.Timedelta(days=horizon)
        
        train_mask = df['date'] < split_date
        test_mask = df['date'] >= split_date
        hist_mask = (df['date'] < split_date) & (df['date'] >= split_date - pd.Timedelta(days=60))
        
        num_cols = [
            'dayofweek', 'month', 'day', 'is_weekend', 'cluster', 'store_type', 
            'dcoilwtico', 'is_holiday', 'lag_base', 'lag_base_1', 'lag_base_7', 'rolling_mean_7'
        ]
        features = ['store_nbr', 'family'] + num_cols
        
        # Выделение признаков и таргета
        X_train, y_train = df.loc[train_mask, features].copy(), df.loc[train_mask, 'sales']
        X_test, y_test = df.loc[test_mask, features].copy(), df.loc[test_mask, 'sales']
        
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        # Шаблон для сохранения истории 
        df_history = df.loc[hist_mask, ['date', 'store_nbr', 'family', 'sales']].copy()
        df_history['predictions'] = np.nan
        df_history['data_type'] = 'history'
        df_history['horizon'] = horizon
        
        for model_type in models:
            print(f"  Обучение модели {model_type}...")
            importance_vector = np.zeros(len(features))
            
            if model_type == "XGBoost":
                # Копии датасетов с приведением к типу category для XGBoost
                X_tr_xgb = X_train.copy()
                X_te_xgb = X_test.copy()
                X_tr_xgb['store_nbr'] = X_tr_xgb['store_nbr'].astype('category')
                X_tr_xgb['family'] = X_tr_xgb['family'].astype('category')
                X_te_xgb['store_nbr'] = X_te_xgb['store_nbr'].astype('category')
                X_te_xgb['family'] = X_te_xgb['family'].astype('category')
                
                model = XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, enable_categorical=True, random_state=42, n_jobs=-1)
                model.fit(X_tr_xgb, y_train_log, eval_set=[(X_te_xgb, y_test_log)], verbose=False)
                preds_log = model.predict(X_te_xgb)
                importance_vector = model.feature_importances_
                
            elif model_type == "CatBoost":
                # Копии датасетов со строковым типом для CatBoost
                X_tr_cat = X_train.copy()
                X_te_cat = X_test.copy()
                X_tr_cat['store_nbr'] = X_tr_cat['store_nbr'].astype(str)
                X_tr_cat['family'] = X_tr_cat['family'].astype(str)
                X_te_cat['store_nbr'] = X_te_cat['store_nbr'].astype(str)
                X_te_cat['family'] = X_te_cat['family'].astype(str)
                
                model = CatBoostRegressor(iterations=150, learning_rate=0.06, depth=6, random_seed=42, verbose=False)
                model.fit(X_tr_cat, y_train_log, cat_features=['store_nbr', 'family'])
                preds_log = model.predict(X_te_cat)
                
                cb_imp = np.array(model.get_feature_importance())
                importance_vector = cb_imp / (cb_imp.sum() if cb_imp.sum() > 0 else 1)
                
            elif model_type == "LSTM":
                scaler = StandardScaler()
                X_train_num_sc = scaler.fit_transform(df.loc[train_mask, num_cols])
                X_test_num_sc = scaler.transform(df.loc[test_mask, num_cols])
                
                # Подготовка тензоров PyTorch
                X_tr_store = torch.tensor(df.loc[train_mask, 'store_code'].values, dtype=torch.long).unsqueeze(1)
                X_tr_family = torch.tensor(df.loc[train_mask, 'family_code'].values, dtype=torch.long).unsqueeze(1)
                X_tr_num = torch.tensor(X_train_num_sc, dtype=torch.float32).unsqueeze(1)
                y_tr_tensor = torch.tensor(y_train_log.values, dtype=torch.float32).unsqueeze(1)
                
                X_te_store = torch.tensor(df.loc[test_mask, 'store_code'].values, dtype=torch.long).unsqueeze(1)
                X_te_family = torch.tensor(df.loc[test_mask, 'family_code'].values, dtype=torch.long).unsqueeze(1)
                X_te_num = torch.tensor(X_test_num_sc, dtype=torch.float32).unsqueeze(1)
                
                model = SalesLSTMWithEmbeddings(num_stores, num_families, store_dim=8, family_dim=6, num_numerical=len(num_cols))
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                loss_fn = nn.MSELoss()
                
                batch_size = 4096
                for epoch in range(5):
                    model.train()
                    for i in range(0, len(X_tr_store), batch_size):
                        optimizer.zero_grad()
                        loss = loss_fn(model(X_tr_store[i:i+batch_size], X_tr_family[i:i+batch_size], X_tr_num[i:i+batch_size]), y_tr_tensor[i:i+batch_size])
                        loss.backward()
                        optimizer.step()
                        
                model.eval()
                preds_list = []
                with torch.no_grad():
                    for i in range(0, len(X_te_store), batch_size):
                        preds_b = model(X_te_store[i:i+batch_size], X_te_family[i:i+batch_size], X_te_num[i:i+batch_size]).squeeze(-1).numpy()
                        preds_list.extend(preds_b)
                preds_log = np.array(preds_list)

            # Обратное преобразование логарифма таргета
            preds = np.maximum(np.expm1(preds_log), 0)
            
            # Сборка прогнозной таблицы для текущей модели
            df_test = df.loc[test_mask, ['date', 'store_nbr', 'family', 'sales']].copy()
            df_test['predictions'] = preds.astype('float32')
            df_test['data_type'] = 'test'
            df_test['horizon'] = horizon
            df_test['model_type'] = model_type
            
            df_curr_hist = df_history.copy()
            df_curr_hist['predictions'] = df_curr_hist['predictions'].astype('float32')
            df_curr_hist['model_type'] = model_type
            
            master_storage.append(df_test)
            master_storage.append(df_curr_hist)
            
            # Сборка важности фичей
            importance_storage.append(pd.DataFrame({
                'horizon': horizon,
                'model_type': model_type,
                'feature': features,
                'importance': importance_vector.astype('float32')
            }))
            
    
    final_df = pd.concat(master_storage, ignore_index=True)
    final_df['store_nbr'] = final_df['store_nbr'].astype(int)
    final_df['family'] = final_df['family'].astype(str)
    final_df.to_parquet('precomputed_predictions.parquet', index=False)
    
    final_imp_df = pd.concat(importance_storage, ignore_index=True)
    final_imp_df.to_parquet('feature_importance.parquet', index=False)
    print("Готово")

if __name__ == "__main__":
    main()
