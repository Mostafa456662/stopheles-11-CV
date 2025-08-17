from tensorflow import keras as k

class PredictionCallback(k.callbacks.Callback):
    def __init__(self, X_sample, y_sample, sample_size=10):
        super().__init__()
        self.X_sample = X_sample[:sample_size]
        self.y_sample = y_sample[:sample_size]
        self.sample_size = sample_size
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- EPOCH {epoch + 1} PREDICTIONS ---")
        predictions = self.model.predict(self.X_sample, verbose=0)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        
        print(f"{'Sample':<6} {'True':<6} {'Pred':<6} {'Prob':<8} {'Match':<6}")
        print("-" * 35)
        
        correct = 0
        for i in range(self.sample_size):
            true_label = "Pos" if self.y_sample[i] == 1 else "Neg"
            pred_label = "Pos" if predicted_labels[i] == 1 else "Neg"
            prob = predictions[i][0]
            match = "✓" if self.y_sample[i] == predicted_labels[i] else "✗"
            
            if self.y_sample[i] == predicted_labels[i]:
                correct += 1
                
            print(f"{i+1:<6} {true_label:<6} {pred_label:<6} {prob:<8.4f} {match:<6}")
        
        accuracy = correct / self.sample_size
        print(f"Sample Accuracy: {accuracy:.4f}\n")