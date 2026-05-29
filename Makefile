.PHONY: up down logs test
up:
	docker compose --env-file .env up -d
	@echo "Waiting for services..."
	@sleep 3
	@docker compose ps
down:
	docker compose down
logs:
	docker compose logs -f
test:
	@cd tests && python -m pytest smoke/ -v