from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:mysql123@164.52.200.229/QAdashboard'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Client(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    email_address = db.Column(db.String(100), nullable=False)

class Agent(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(36), db.ForeignKey('client.id', name='fk_interaction_client'), nullable=False)
    agent_id = db.Column(db.String(36), db.ForeignKey('agent.id', name='fk_interaction_agent'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Integer, nullable=False)
    escalation = db.Column(db.Boolean, nullable=False)
    channel = db.Column(db.String(20), nullable=False)
    audio_file_path = db.Column(db.String(200), nullable=False)
    transcription_text_path = db.Column(db.String(200), nullable=False)
    customer_stated_purpose = db.Column(db.String(500), nullable=False)
    purpose_clarity_score = db.Column(db.Integer, nullable=False)
    overall_interaction_score = db.Column(db.Integer, nullable=False)
    overall_sentiment = db.Column(db.String(10), nullable=False)
    std_opening_score = db.Column(db.Integer, nullable=False)
    std_closing_score = db.Column(db.Integer, nullable=False)
    ownership_score = db.Column(db.Integer, nullable=False)
    tone_score = db.Column(db.Integer, nullable=False)
    probing_verifications_score = db.Column(db.Integer, nullable=False)

    client = db.relationship('Client', backref=db.backref('interactions', lazy=True))
    agent = db.relationship('Agent', backref=db.backref('interactions', lazy=True))

db.create_all()

@app.route('/interactions', methods=['GET'])
def get_interactions():
    try:
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Interaction")
        total_calls = cursor.fetchone()[0]
        logging.info(f"Total calls fetched from database: {total_calls}")
        return jsonify({"total_calls": total_calls})
    except Exception as e:
        logging.error(f"Error fetching interactions: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/trends', methods=['GET'])
def get_trends():
    period = request.args.get('period', 'week')
    now = datetime.now()
    
    if period == 'month':
        start_date = now - timedelta(days=30)
    else:
        start_date = now - timedelta(days=7)

    interactions = Interaction.query.filter(Interaction.start_time >= start_date).all()
    output = []

    for interaction in interactions:
        trend_data = {
            'call_id': interaction.id,
            'client_id': interaction.client_id,
            'client_name': interaction.client.name,
            'agent_id': interaction.agent_id,
            'agent_name': interaction.agent.name,
            'medium_of_contact': interaction.channel,
            'duration': interaction.duration
        }
        output.append(trend_data)

    return jsonify({'trends': output})


@app.route('/interactions/client', methods=['GET'])
def get_client_interactions():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({'error': 'Client ID is required'}), 400

    interactions = Interaction.query.filter_by(client_id=client_id).all()
    output = []

    for interaction in interactions:
        interaction_data = {
            'id': interaction.id,
            'client_id': interaction.client_id,
            'agent_id': interaction.agent_id,
            'start_time': interaction.start_time,
            'end_time': interaction.end_time,
            'duration': interaction.duration,
            'escalation': interaction.escalation,
            'channel': interaction.channel,
            'audio_file_path': interaction.audio_file_path,
            'transcription_text_path': interaction.transcription_text_path,
            'customer_stated_purpose': interaction.customer_stated_purpose,
            'purpose_clarity_score': interaction.purpose_clarity_score,
            'overall_interaction_score': interaction.overall_interaction_score,
            'overall_sentiment': interaction.overall_sentiment,
            'std_opening_score': interaction.std_opening_score,
            'std_closing_score': interaction.std_closing_score,
            'ownership_score': interaction.ownership_score,
            'tone_score': interaction.tone_score,
            'probing_verifications_score': interaction.probing_verifications_score,
            'client': {
                'id': interaction.client.id,
                'name': interaction.client.name,
                'address': interaction.client.address,
                'phone_number': interaction.client.phone_number,
                'email_address': interaction.client.email_address
            },
            'agent': {
                'id': interaction.agent.id,
                'name': interaction.agent.name,
                'role': interaction.agent.role
            }
        }
        output.append(interaction_data)

    return jsonify({'interactions': output})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='164.52.200.229', port=9486)
