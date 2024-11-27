pipeline {
    agent any
    environment {
        APP_URI = credentials('app-uri') // Nom du credential pour APP_URI
        AWS_ACCESS_KEY_ID = credentials('aws-access-key-id') // Credential AWS Access Key
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-access-key') // Credential AWS Secret Key
        BACKEND_STORE_URI = credentials('backend-store-uri') // Credential Backend Store URI
        ARTIFACT_STORE_URI = credentials('artifact-store-uri') // Credential Artifact Store URI
        POSTMARK_API_KEY = credentials('postmark-api-key') // Credential Postmark API Key
        MLFLOW_EXPERIMENT_NAME = credentials('mlflow-experiment-name') // Credential pour le nom de l'expérience
    }
    stages {
        stage('Checkout') {
            steps {
                // Jenkins ne gère que l'accès GitHub ici
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                dir('CICD/mlflow') {
                    script {
                        sh """
                        docker build -t mlflow-image --build-arg REPO_URL=https://github.com/username/project-name.git .
                        """
                    }
                }
            }
        }

        stage('Run Training Script') {
            steps {
                script {
                    sh """
                    docker run --env-file CICD/.env -p 4000:4000 mlflow-image python /home/app/train.py
                    """
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    def test_result = sh(script: """
                    docker run --env-file CICD/.env \
                        mlflow-image \
                        pytest --maxfail=1 --disable-warnings -q /home/app/tests
                    """, returnStatus: true)

                    if (test_result != 0) {
                        echo "Tests failed. Sending alert email..."
                        sh """
                        docker run --env-file CICD/.env \
                            mlflow-image \
                            python /home/app/send_email.py "Tests Failed" "Les tests ont échoué. Vérifiez les logs Jenkins."
                        """
                        error "Tests failed. Pipeline stopped."
                    } else {
                        echo "All tests passed successfully."
                    }
                }
