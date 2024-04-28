build:
	docker build . -t yadro_hack

run:
	docker run --name yadro_hack yadro_hack:latest
	docker cp yadro_hack:/app/solution/solution.csv solution/solution.csv
	docker stop /yadro_hack
	docker rm /yadro_hack