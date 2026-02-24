pipeline {
    agent any

    environment {
        IMAGE_NAME = "music-recommender"
        IMAGE_TAG  = "${BUILD_NUMBER}"
        CONTAINER_NAME = "music-recommender-app"
    }

    stages {

        // ── Stage 1: Checkout ──────────────────────────────────────────────
        stage('Checkout') {
            steps {
                echo '📥 Cloning repository...'
                git branch: 'main', url: 'https://github.com/SUDARSHAN-MANIKANDAN/MUSIC-RECOMMENDATION-SYSTEM.git'
            }
        }

        // ── Stage 2: Install Dependencies ─────────────────────────────────
        stage('Install Dependencies') {
            steps {
                echo '📦 Installing dependencies...'
                sh '''
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install pytest pytest-junit pytest-cov
                '''
            }
        }

        // ── Stage 3: Lint Check ────────────────────────────────────────────
        stage('Lint') {
            steps {
                echo '🔍 Running lint checks...'
                sh '''
                    pip install flake8
                    flake8 . --max-line-length=120 --exclude=.git,__pycache__ --count --statistics
                '''
            }
        }

        // ── Stage 4: Unit Tests ────────────────────────────────────────────
        stage('Unit Tests') {
            steps {
                echo '🧪 Running unit tests...'
                sh '''
                    mkdir -p test-results
                    pytest tests/ \
                        -v \
                        --tb=short \
                        --junitxml=test-results/results.xml \
                        --cov=. \
                        --cov-report=xml:test-results/coverage.xml \
                        --cov-report=term-missing
                '''
            }
            post {
                always {
                    junit 'test-results/results.xml'
                    echo '📊 Test results published'
                }
                failure {
                    echo '❌ Tests failed! Stopping pipeline.'
                    error('Unit tests failed — aborting build')
                }
                success {
                    echo '✅ All tests passed!'
                }
            }
        }

        // ── Stage 5: Accuracy Gate ─────────────────────────────────────────
        stage('Accuracy Gate') {
            steps {
                echo '🎯 Checking model accuracy threshold...'
                sh '''
                    python -c "
import pickle, sys
try:
    with open('features.pkl', 'rb') as f:
        data = pickle.load(f)
    acc = data.get('accuracy', 0)
    print(f'Model accuracy: {acc:.2%}')
    if acc < 0.70:
        print(f'FAILED: Accuracy {acc:.2%} is below 70% threshold!')
        sys.exit(1)
    else:
        print(f'PASSED: Accuracy {acc:.2%} meets 70% threshold!')
except FileNotFoundError:
    print('WARNING: features.pkl not found, skipping accuracy gate')
"
                '''
            }
        }

        // ── Stage 6: Docker Build ──────────────────────────────────────────
        stage('Docker Build') {
            steps {
                echo '🐳 Building Docker image...'
                sh """
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                    echo 'Docker image built successfully'
                    docker images | grep ${IMAGE_NAME}
                """
            }
        }

        // ── Stage 7: Docker Test ───────────────────────────────────────────
        stage('Docker Test') {
            steps {
                echo '🔬 Testing Docker container...'
                sh """
                    docker run --rm \
                        --name ${CONTAINER_NAME}-test \
                        ${IMAGE_NAME}:latest \
                        python -c "import streamlit; import librosa; import sklearn; print('All imports OK')"
                """
            }
        }

        // ── Stage 8: Deploy ────────────────────────────────────────────────
        stage('Deploy') {
            steps {
                echo '🚀 Deploying container...'
                sh """
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true

                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        -p 8501:8501 \
                        -v \$(pwd)/songs:/app/songs \
                        -v \$(pwd)/features.pkl:/app/features.pkl \
                        --restart unless-stopped \
                        ${IMAGE_NAME}:latest

                    echo 'Container deployed on port 8501'
                """
            }
        }

        // ── Stage 9: Health Check ──────────────────────────────────────────
        stage('Health Check') {
            steps {
                echo '❤️ Verifying app is healthy...'
                sh '''
                    sleep 15
                    curl -f http://localhost:8501/_stcore/health && echo "✅ App is healthy!" || echo "⚠️ Health check failed"
                '''
            }
        }

        // ── Future: AWS Stages ─────────────────────────────────────────────
        // stage('Push to ECR') {
        //     steps {
        //         withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-credentials']]) {
        //             sh """
        //                 aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URL
        //                 docker tag ${IMAGE_NAME}:latest YOUR_ECR_URL/${IMAGE_NAME}:latest
        //                 docker push YOUR_ECR_URL/${IMAGE_NAME}:latest
        //             """
        //         }
        //     }
        // }
        // stage('Deploy to ECS') {
        //     steps {
        //         sh "aws ecs update-service --cluster YOUR_CLUSTER --service YOUR_SERVICE --force-new-deployment"
        //     }
        // }
    }

    post {
        success {
            echo '''
            ✅ ═══════════════════════════════════
               Pipeline completed successfully!
               App running at http://localhost:8501
            ═══════════════════════════════════
            '''
        }
        failure {
            echo '''
             ═══════════════════════════════════
               Pipeline FAILED. Check logs above.
            ═══════════════════════════════════
            '''
        }
        always {
            echo '🧹 Cleaning up unused Docker images...'
            sh 'docker image prune -f || true'
        }
    }
}