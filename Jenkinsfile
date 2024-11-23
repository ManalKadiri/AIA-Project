pipeline {
    agent any
    environment {
        // Injecter les credentials Jenkins
        AWS_ACCESS_KEY_ID = credentials('aws-access-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-access-key')
        BACKEND_STORE_URI = credentials('backend-store-uri')
        ARTIFACT_STORE_URI = credentials('artifact-store-uri')
        APP_URI = credentials('app-uri')
    }
    stages {
        stage('Checkout') {
            steps {
                // Cloner le code source depuis GitHub
                checkout scm
            }
        }
        stage('Build Docker Image') {
            steps {
                dir('mlflow') {
                    sh 'docker build -t mlflow-image .'
                }
            }
        }
        stage('Generate .env File') {
            steps {
                script {
                    // Créez dynamiquement le fichier .env dans le dossier mlflow
                    writeFile file: 'mlflow/.env', text: """
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
BACKEND_STORE_URI=${BACKEND_STORE_URI}
ARTIFACT_STORE_URI=${ARTIFACT_STORE_URI}
APP_URI=${APP_URI}
"""
                }
            }
        }
        stage('Run Training Script') {
            steps {
                dir('mlflow') {
                    sh """
                    docker run --env-file .env -p 4000:4000 -v "\$(pwd):/home/app" mlflow-image python train.py
                    """
                }
            }
        }
        stage('Run Tests') {
            steps {
                dir('mlflow') {
                    script {
                        // Lancer les tests dans le conteneur Docker
                        sh """
                        docker run --env-file .env \
                            -v "\$(pwd)/tests:/home/app/tests" \
                            -v "\$(pwd)/mlflow:/home/app/mlflow" \
                            mlflow-image pytest /home/app/tests --disable-warnings
                        """
                    }
                }
            }
        }
        stage('Post-build Cleanup') {
            steps {
                echo "Cleaning up..."
                sh 'docker rmi mlflow-image || true'
            }
        }
    }
}



