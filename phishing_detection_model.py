"""
Phishing Email Detection Model
Uses scikit-learn to classify emails as Phishing or Safe
Features: Text analysis, URL analysis, keyword detection, and linguistic patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PhishingDetectionModel:
    """Machine Learning model for phishing email detection"""
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.scaler = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_data(self, filepath='phishing_emails.csv'):
        """Load email dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Dataset loaded: {len(df)} emails")
            print(f"  - Phishing: {len(df[df['phishing'] == 1])}")
            print(f"  - Legitimate: {len(df[df['phishing'] == 0])}")
            return df
        except FileNotFoundError:
            print(f"✗ Error: {filepath} not found")
            print("  Run: python create_dataset.py")
            return None
    
    def extract_features(self, df):
        """Extract features from emails"""
        print("\n📊 Extracting Features...")
        
        # Text features using TF-IDF
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = self.tfidf.fit_transform(df['email'].astype(str))
        
        # Numerical features
        numerical_features = df[[
            'keyword_count',
            'exclamation_marks',
            'urgent_keywords',
            'suspicious_urls',
            'misspelled_words',
            'all_caps_words'
        ]].values
        
        # Combine features
        combined_features = np.hstack([
            text_features.toarray(),
            numerical_features
        ])
        
        # Scale numerical features
        self.scaler = StandardScaler()
        numerical_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine scaled features
        X = np.hstack([text_features.toarray(), numerical_scaled])
        
        print(f"✓ Features extracted: {X.shape[1]} total features")
        print(f"  - TF-IDF text features: {text_features.shape[1]}")
        print(f"  - Numerical features: 6")
        
        return X, df['phishing'].values
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\n✓ Data split:")
        print(f"  - Training set: {len(self.X_train)} emails")
        print(f"  - Testing set: {len(self.X_test)} emails")
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n🧠 Training Logistic Regression...")
        self.model = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        print("✓ Logistic Regression trained")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        print("✓ Random Forest trained")
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n⚡ Training Gradient Boosting...")
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        print("✓ Gradient Boosting trained")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📈 Model Evaluation")
        print("=" * 50)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print(f"\n✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"✓ F1-Score:  {f1:.4f}")
        print(f"✓ ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n📊 Confusion Matrix:")
        print(f"  True Negatives:  {tn} (Correctly identified Legitimate)")
        print(f"  False Positives: {fp} (Legitimate marked as Phishing)")
        print(f"  False Negatives: {fn} (Phishing marked as Legitimate) ⚠️")
        print(f"  True Positives:  {tp} (Correctly identified Phishing)")
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['Legitimate', 'Phishing']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def cross_validate(self, cv=5):
        """Perform cross-validation"""
        print(f"\n🔄 Cross-Validation ({cv}-Fold)...")
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - Phishing Detection Model', fontsize=14, fontweight='bold')
        
        # Add accuracy metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        plt.text(0.5, -0.25, f'Accuracy: {accuracy:.2%}', 
                ha='center', transform=plt.gca().transAxes, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Phishing Detection Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved as 'roc_curve.png'")
        plt.show()
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature importances
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            # Feature names
            feature_names = list(self.tfidf.get_feature_names_out()) + [
                'Keyword Count', 'Exclamation Marks', 'Urgent Keywords',
                'Suspicious URLs', 'Misspelled Words', 'All Caps Words'
            ]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
            plt.title(f'Top {top_n} Feature Importance - Phishing Detection', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance saved as 'feature_importance.png'")
            plt.show()
    
    def predict_email(self, email_text, keyword_count=0, exclamation_marks=0,
                     urgent_keywords=0, suspicious_urls=0, misspelled_words=0, all_caps_words=0):
        """Predict if an email is phishing or legitimate"""
        # Vectorize text
        text_vector = self.tfidf.transform([email_text]).toarray()
        
        # Create feature vector
        features = np.array([[keyword_count, exclamation_marks, urgent_keywords,
                            suspicious_urls, misspelled_words, all_caps_words]])
        features_scaled = self.scaler.transform(features)
        
        # Combine features
        X = np.hstack([text_vector, features_scaled])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return prediction, probability
    
    def interactive_test(self):
        """Interactive testing of the model"""
        print("\n" + "="*60)
        print("🧪 INTERACTIVE EMAIL TESTING")
        print("="*60)
        
        test_emails = [
            {
                'text': 'Hi, your package has arrived. Track it at amazon.com',
                'keyword_count': 0,
                'exclamation_marks': 0,
                'urgent_keywords': 0,
                'suspicious_urls': 0,
                'misspelled_words': 0,
                'all_caps_words': 0
            },
            {
                'text': 'URGENT!!! Confirm your PayPal details NOW or account LOCKED!!!',
                'keyword_count': 3,
                'exclamation_marks': 4,
                'urgent_keywords': 3,
                'suspicious_urls': 1,
                'misspelled_words': 0,
                'all_caps_words': 3
            },
            {
                'text': 'Team meeting tomorrow at 2 PM. Please review the agenda.',
                'keyword_count': 0,
                'exclamation_marks': 0,
                'urgent_keywords': 0,
                'suspicious_urls': 0,
                'misspelled_words': 0,
                'all_caps_words': 0
            }
        ]
        
        for i, email in enumerate(test_emails, 1):
            print(f"\nTest Email {i}:")
            print(f"Text: {email['text']}")
            
            prediction, prob = self.predict_email(
                email['text'],
                email['keyword_count'],
                email['exclamation_marks'],
                email['urgent_keywords'],
                email['suspicious_urls'],
                email['misspelled_words'],
                email['all_caps_words']
            )
            
            label = 'PHISHING ⚠️' if prediction == 1 else 'LEGITIMATE ✓'
            phishing_prob = prob[1] * 100
            safe_prob = prob[0] * 100
            
            print(f"Prediction: {label}")
            print(f"Confidence: Phishing {phishing_prob:.2f}% | Safe {safe_prob:.2f}%")
            print("-" * 60)

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🚀 PHISHING EMAIL DETECTION MODEL")
    print("="*60)
    
    # Initialize model
    model = PhishingDetectionModel()
    
    # Load data
    df = model.load_data('phishing_emails.csv')
    if df is None:
        print("\n⚠️  Creating dataset first...")
        import subprocess
        subprocess.run(['python', 'create_dataset.py'])
        df = model.load_data('phishing_emails.csv')
    
    # Extract features
    X, y = model.extract_features(df)
    
    # Split data
    model.split_data(X, y)
    
    # Train models and compare
    print("\n" + "="*60)
    print("🤖 COMPARING MODELS")
    print("="*60)
    
    models_to_train = [
        ('Logistic Regression', model.train_logistic_regression),
        ('Random Forest', model.train_random_forest),
        ('Gradient Boosting', model.train_gradient_boosting)
    ]
    
    results = {}
    
    for model_name, train_func in models_to_train:
        train_func()
        metrics = model.evaluate_model()
        results[model_name] = metrics
        model.cross_validate()
    
    # Select best model (using accuracy)
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Re-train best model
    if best_model[0] == 'Logistic Regression':
        model.train_logistic_regression()
    elif best_model[0] == 'Random Forest':
        model.train_random_forest()
    else:
        model.train_gradient_boosting()
    
    # Generate visualizations
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    model.plot_confusion_matrix()
    model.plot_roc_curve()
    
    if best_model[0] != 'Logistic Regression':
        model.plot_feature_importance()
    
    # Interactive testing
    model.interactive_test()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png (if tree-based model)")

if __name__ == "__main__":
    main()
