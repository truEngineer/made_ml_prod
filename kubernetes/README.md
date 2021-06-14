# Kubernetes

## kubectl + minicube

- install [kubectl](https://kubernetes.io/docs/tasks/tools/) 
- install [minicube](https://minikube.sigs.k8s.io/docs/start/)

Check:

```bash
user$ minikube start
user$ kubectl cluster-info
```

## online-inference

Create pod:

```bash
user$ kubectl apply -f online-inference-pod.yaml
user$ kubectl get pods
```
