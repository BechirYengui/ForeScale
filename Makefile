# ForeScale -- one-command local demo.
#
# Quick path (no Kubernetes needed): `make demo-sim` -> results/comparison.png
# Full path (kind required):         `make up` then `make demo`
#
# Variables can be overridden, e.g.  make up CLUSTER=foo
CLUSTER     ?= forescale
NS          ?= forescale
KIND        ?= kind
KUBECTL     ?= kubectl
PYTHON      ?= python
VENV        ?= .venv
REGISTRY    ?= forescale
DAY_SECONDS ?= 600

IMAGES := inference-api load-generator controller

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Local dev / tests (no cluster)
# --------------------------------------------------------------------------- #
.PHONY: venv
venv: ## Create the dev virtualenv and install deps
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip \
		&& pip install -e ./libs && pip install -r requirements-dev.txt

.PHONY: test
test: ## Run unit tests
	. $(VENV)/bin/activate && pytest

.PHONY: lint
lint: ## Run ruff
	. $(VENV)/bin/activate && ruff check .

.PHONY: train
train: ## Train the forecaster -> forecaster.pkl (prints MAE vs baseline)
	. $(VENV)/bin/activate && $(PYTHON) -m forecaster.train --out forecaster.pkl

.PHONY: preview
preview: ## Plot the synthetic traffic curve -> results/traffic_preview.png
	. $(VENV)/bin/activate && $(PYTHON) -m experiments.preview_traffic

.PHONY: demo-sim
demo-sim: ## Offline comparison (no cluster) -> results/comparison.png + results.md
	. $(VENV)/bin/activate && $(PYTHON) -m experiments.run_comparison --mode sim \
		--day-seconds $(DAY_SECONDS)

# --------------------------------------------------------------------------- #
# Kubernetes (kind) demo
# --------------------------------------------------------------------------- #
.PHONY: up
up: cluster images load deploy metrics-server ## Create cluster, build/load images, deploy everything
	@echo ">> ForeScale is up. Run 'make demo' (full) or 'make demo-sim' (offline)."

.PHONY: cluster
cluster: ## Create the kind cluster
	$(KIND) get clusters | grep -qx $(CLUSTER) || \
		$(KIND) create cluster --name $(CLUSTER) --config k8s/kind-config.yaml

.PHONY: images
images: ## Build the three Docker images (repo root as context)
	docker build -f services/inference-api/Dockerfile -t $(REGISTRY)/inference-api:latest services/inference-api
	docker build -f services/load-generator/Dockerfile -t $(REGISTRY)/load-generator:latest .
	docker build -f services/forescale-controller/Dockerfile -t $(REGISTRY)/controller:latest .

.PHONY: load
load: ## Load images into the kind cluster
	$(KIND) load docker-image $(REGISTRY)/inference-api:latest --name $(CLUSTER)
	$(KIND) load docker-image $(REGISTRY)/load-generator:latest --name $(CLUSTER)
	$(KIND) load docker-image $(REGISTRY)/controller:latest --name $(CLUSTER)

.PHONY: metrics-server
metrics-server: ## Install metrics-server (needed by the reactive HPA)
	$(KUBECTL) apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	# kind's kubelet serving cert is self-signed: allow insecure TLS for the demo.
	$(KUBECTL) -n kube-system patch deployment metrics-server --type='json' \
		-p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
	$(KUBECTL) -n kube-system rollout status deployment/metrics-server --timeout=120s

.PHONY: deploy
deploy: ## Apply all manifests + create Grafana provisioning ConfigMaps
	$(KUBECTL) apply -f k8s/00-namespace.yaml
	$(KUBECTL) apply -f k8s/10-configmap.yaml
	-$(KUBECTL) -n $(NS) create configmap grafana-provisioning \
		--from-file=observability/grafana/datasource.yaml \
		--from-file=observability/grafana/dashboards.yaml \
		--dry-run=client -o yaml | $(KUBECTL) apply -f -
	-$(KUBECTL) -n $(NS) create configmap grafana-dashboards \
		--from-file=observability/grafana/dashboard.json \
		--dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) apply -f k8s/20-inference-api.yaml
	$(KUBECTL) apply -f k8s/30-controller-rbac.yaml
	$(KUBECTL) apply -f k8s/31-controller.yaml
	$(KUBECTL) apply -f k8s/60-prometheus.yaml
	$(KUBECTL) apply -f k8s/70-grafana.yaml
	$(KUBECTL) -n $(NS) rollout status deployment/inference-api --timeout=180s
	$(KUBECTL) -n $(NS) rollout status deployment/prometheus --timeout=180s

.PHONY: demo
demo: ## Full reactive-vs-predictive run on the cluster -> results/
	. $(VENV)/bin/activate && $(PYTHON) -m experiments.run_comparison --mode k8s \
		--day-seconds $(DAY_SECONDS)

.PHONY: demo-reactive
demo-reactive: ## Reactive (HPA) only, on the cluster
	$(KUBECTL) -n $(NS) scale deployment/forescale-controller --replicas=0
	$(KUBECTL) apply -f k8s/40-hpa.yaml
	@echo ">> Reactive mode active (HPA on, ForeScale off). Drive load with the load-generator."

.PHONY: demo-predictive
demo-predictive: ## Predictive (ForeScale) only, on the cluster
	-$(KUBECTL) -n $(NS) delete hpa inference-api --ignore-not-found
	$(KUBECTL) -n $(NS) scale deployment/forescale-controller --replicas=1
	@echo ">> Predictive mode active (ForeScale on, HPA removed)."

.PHONY: grafana
grafana: ## Port-forward Grafana to http://localhost:3000 (admin/admin)
	$(KUBECTL) -n $(NS) port-forward svc/grafana 3000:3000

.PHONY: prometheus
prometheus: ## Port-forward Prometheus to http://localhost:9090
	$(KUBECTL) -n $(NS) port-forward svc/prometheus 9090:9090

.PHONY: down
down: ## Delete the kind cluster
	$(KIND) delete cluster --name $(CLUSTER)
