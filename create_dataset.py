import pandas as pd
import numpy as np

# Create a dataset of phishing and legitimate emails
def create_email_dataset():
    """Create a dataset of phishing and legitimate emails with labels"""
    
    phishing_emails = [
        {
            'email': 'Urgent: Verify your PayPal account now! Click here immediately to confirm your password and prevent account suspension.',
            'urls': 'paypal-verify-123.tk',
            'keyword_count': 4,
            'exclamation_marks': 3,
            'urgent_keywords': 3,
            'suspicious_urls': 1,
            'misspelled_words': 1,
            'all_caps_words': 2,
            'phishing': 1
        },
        {
            'email': 'CONFIRM YOUR BANKING DETAILS NOW!!! Your account will be closed in 24 hours. Update payment method immediately!!!',
            'urls': 'bank-security-update-verify.xyz',
            'keyword_count': 5,
            'exclamation_marks': 5,
            'urgent_keywords': 4,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 5,
            'phishing': 1
        },
        {
            'email': 'Amazon Security Alert: Unusual activity detected. Verify your identity by clicking the link below to secure your account.',
            'urls': 'amaz0n-security-check.ru',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'Congratulations! You have won a FREE gift card worth $500. Click here to claim your prize now before it expires!',
            'urls': 'claim-prize-gift.tk',
            'keyword_count': 4,
            'exclamation_marks': 2,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 2,
            'phishing': 1
        },
        {
            'email': 'Action Required: Your Apple ID will be locked in 48 hours. Verify your account details immediately by clicking below.',
            'urls': 'apple-id-verification-secure.pw',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 3,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'Important Notice: Your Google account requires immediate verification. Click link to confirm password and protect your account.',
            'urls': 'g00gle-verify-now.net',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'Claim your $1000 Target Gift Card NOW!!! Limited time offer. Click here to redeem your card immediately!!!',
            'urls': 'target-giftcard-claim-here.xyz',
            'keyword_count': 5,
            'exclamation_marks': 4,
            'urgent_keywords': 3,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 3,
            'phishing': 1
        },
        {
            'email': 'URGENT: Your credit card will be BLOCKED! Confirm your banking details immediately. Do not ignore this message!!!',
            'urls': 'bank-security-verify-urgent.ru',
            'keyword_count': 5,
            'exclamation_marks': 4,
            'urgent_keywords': 4,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 4,
            'phishing': 1
        },
        {
            'email': 'PayPal Security: Unusual login detected. Confirm your account by clicking the link below within 24 hours.',
            'urls': 'paypal-security-confirm.tk',
            'keyword_count': 3,
            'exclamation_marks': 0,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'You\'ve inherited $2,500,000!!! Reply to this email with your banking details to claim your inheritance immediately!!!',
            'urls': 'inheritance-claim-verify.tk',
            'keyword_count': 4,
            'exclamation_marks': 5,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 2,
            'all_caps_words': 2,
            'phishing': 1
        },
        {
            'email': 'Microsoft Account: Your account has been locked due to suspicious activity. Click here to unlock immediately.',
            'urls': 'microsoft-account-verify-secure.xyz',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'Western Union Alert: Someone tried to transfer money from your account! Verify identity immediately at confirm-transfer.net',
            'urls': 'confirm-transfer.net',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 1
        },
        {
            'email': 'FedEx Package Delivery: Your package could not be delivered. Update your address at fedex-delivery-track.tk now!',
            'urls': 'fedex-delivery-track.tk',
            'keyword_count': 3,
            'exclamation_marks': 1,
            'urgent_keywords': 1,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 1
        },
        {
            'email': 'IRS Tax Alert: You have a pending tax refund! Click below to claim your refund of $3,420 now!!!',
            'urls': 'irs-tax-refund-claim.tk',
            'keyword_count': 4,
            'exclamation_marks': 3,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        },
        {
            'email': 'Netflix Account Suspended: Your payment method declined. Update your billing information immediately at netflix-update-billing.xyz',
            'urls': 'netflix-update-billing.xyz',
            'keyword_count': 4,
            'exclamation_marks': 1,
            'urgent_keywords': 2,
            'suspicious_urls': 1,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 1
        }
    ]
    
    legitimate_emails = [
        {
            'email': 'Hi John, Thanks for your order. Your package will arrive tomorrow. You can track it at amazon.com with your order number.',
            'urls': 'amazon.com',
            'keyword_count': 1,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Meeting reminder: Team sync tomorrow at 2 PM in Conference Room B. Please review the agenda before the meeting.',
            'urls': 'company.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Welcome to GitHub! Your account has been created successfully. You can now create and manage repositories.',
            'urls': 'github.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Receipt for your purchase: Order #12345 has been confirmed. Total: $42.99. View details on your account page.',
            'urls': 'shop.example.com',
            'keyword_count': 1,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Your password was changed successfully. If this was not you, please contact our support team immediately.',
            'urls': 'support.example.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Thank you for subscribing to our newsletter. You will receive weekly updates about our latest products and offers.',
            'urls': 'newsletter.example.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Your LinkedIn profile was viewed by a recruiter. Visit LinkedIn to see more details about the company.',
            'urls': 'linkedin.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Project deadline reminder: The Q4 report is due on Friday. Please submit your completed sections to the shared drive.',
            'urls': 'company.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Your flight confirmation: Booking reference ABC123. Your flight departs at 10:30 AM from Terminal 2.',
            'urls': 'airline.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Insurance policy renewal: Your annual premium is $450. Your coverage continues through December 2025.',
            'urls': 'insurance.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Slack notification: You have a new message from the marketing team. Check your Slack workspace for details.',
            'urls': 'slack.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Google Drive share notification: Someone shared a document with you. Check your email to view and edit the file.',
            'urls': 'drive.google.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Bank statement: Your account balance is $5,234.56 as of today. View full statement on our website.',
            'urls': 'bank.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        },
        {
            'email': 'Event invitation: You are invited to our company holiday party on December 15th at 6 PM. Please RSVP by December 1st.',
            'urls': 'company.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 1,
            'phishing': 0
        },
        {
            'email': 'Book confirmation: Your order for "Machine Learning Basics" has been shipped. Tracking number is in your account.',
            'urls': 'bookstore.com',
            'keyword_count': 0,
            'exclamation_marks': 0,
            'urgent_keywords': 0,
            'suspicious_urls': 0,
            'misspelled_words': 0,
            'all_caps_words': 0,
            'phishing': 0
        }
    ]
    
    # Combine datasets
    all_emails = phishing_emails + legitimate_emails
    df = pd.DataFrame(all_emails)
    
    return df

if __name__ == "__main__":
    df = create_email_dataset()
    
    # Save to CSV
    df.to_csv('phishing_emails.csv', index=False)
    print("Dataset created successfully!")
    print(f"Total emails: {len(df)}")
    print(f"Phishing emails: {len(df[df['phishing'] == 1])}")
    print(f"Legitimate emails: {len(df[df['phishing'] == 0])}")
    print("\nDataset saved to 'phishing_emails.csv'")
