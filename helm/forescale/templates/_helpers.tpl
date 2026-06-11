{{/* Common labels applied to every object. */}}
{{- define "forescale.labels" -}}
app.kubernetes.io/part-of: forescale
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/* Fully-qualified image reference for a component. */}}
{{- define "forescale.image" -}}
{{- $root := index . 0 -}}
{{- $name := index . 1 -}}
{{ $root.Values.image.repository }}/{{ $name }}:{{ $root.Values.image.tag }}
{{- end -}}
