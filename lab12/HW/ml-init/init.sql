CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    training_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    training_set_size INTEGER NOT NULL,
    test_mae FLOAT NOT NULL,
    hyperparameters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
