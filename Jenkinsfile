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
                    script {
                        sh '''
                        docker build -t mlflow-image .
                        '''
                    }
                }
            }
        }
        stage('Run Training Script') {
            steps {
                dir('mlflow') {
                    script {
                        sh '''
                        docker run --env-file .env \
                                   -p 4000:4000 \
                                   -v "$(pwd):/home/app" \
                                   mlflow-image python train.py
                        '''
                    }
                }
            }
        }
        stage('Post-build Cleanup') {
            steps {
                echo "Nettoyage après le build..."
                script {
                    sh '''
                    docker rmi mlflow-image || true
                    '''
                }
            }
        }
    }
}

