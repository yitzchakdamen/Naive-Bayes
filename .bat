docker volume create model_files-volume

docker rm -f $(docker ps -aq)
docker rmi -f $(docker images -aq)

docker network create my-network


cd C:\Users\isaac\source\repos\Naive_Bayes\Model_lib
Docker build -f ModelDockerfile -t model-lib__v2 .

cd c:/Users/isaac/source/repos/Naive_Bayes/Model_file_server
docker build -f Dockerfile -t main__v2 .

docker rm main
docker run -d --name main --network my-network -p 8888:8888  main__v2

cd C:\Users\isaac\source\repos\Naive_Bayes\Training_and_testing
docker build -f TrainingTestingDockerfile -t training_and_testing__v2 .

docker rm training_and_testing
docker run -d --name training_and_testing --network my-network -p 8501:8501 training_and_testing__v2

cd C:\Users\isaac\source\repos\Naive_Bayes\Classifier_Server
docker build -f ServerDockerfile -t prediction-server__v2 .

docker rm prediction-server
docker run -d --name prediction-server --network my-network -p 8030:8030  prediction-server__v2

cd C:\Users\isaac\source\repos\Naive_Bayes\Client
docker build -f ClientDockerfile -t client__v2 .

docker rm client
docker run -d --name client --network my-network -p 8601:8601 client__v2


start "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://127.0.0.1:8501/"
start "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://127.0.0.1:8601/"

