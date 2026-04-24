package telemetry

import (
	"context"
	"errors"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/runtime"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

const (
	namespace = "ollama"
)

// RunningModelInfo holds information about a single running model for Prometheus metrics.
type RunningModelInfo struct {
	Name            string // Full model name (e.g., lapo/qwen3.6:35b-a3b-coding-int4)
	Digest          string   // Model digest/hash
	Size            int64  // Total model size in bytes
	SizeVRAM        int64    // VRAM usage in bytes
	ContextLength   int     // Context length
	ExpiresAt       time.Time // When the model expires from memory
	GPUPercent      float64    // Percentage of model on GPU (0-100)
	Format          string        // Model format (gguf, safetensors, etc.)
	Family          string         // Model family (llama, gpt2, etc.)
	Families        []string `json:"families,omitempty"` // List of model families
	ParameterSize   string    // Parameter count (e.g., "7B", "13B")
	QuantizationLevel string   // Quantization level (Q4_0, Q8_0, etc.)
}

type Metrics struct {
	Start              metric.Int64Gauge
	Requests           metric.Int64Counter
	TotalDuration      metric.Float64Counter
	LoadDuration       metric.Float64Counter
	PromptEvalCount    metric.Int64Counter
	PromptEvalDuration metric.Float64Counter
	EvalCount          metric.Int64Counter
	EvalDuration       metric.Float64Counter
	PeakMemory         metric.Int64Gauge

	// Running models metrics (from ollama ps)
	RunningModelsCount metric.Int64Gauge
	ModelSizeBytes     metric.Int64Gauge
	ModelVRAMBytes     metric.Int64Gauge
	ModelContextLength metric.Int64Gauge
	ModelExpiresAt     metric.Int64Gauge
	ModelGPUPercent    metric.Float64Gauge
}

func NewMetrics(meter metric.Meter) *Metrics {
	build, _ := meter.Int64Gauge(
		"ollama_build_info",
		metric.WithDescription("Ollama start date (as Unixtime) and build version."),
		metric.WithUnit("seconds"),
	)

	req, _ := meter.Int64Counter(
		"http_requests_total",
		metric.WithDescription("The total number of requests on the endpoints."),
		metric.WithUnit("requests"),
	)

	totalDuration, _ := meter.Float64Counter(
		"ollama_total_duration_seconds",
		metric.WithDescription("The request total duration in seconds."),
		metric.WithUnit("seconds"),
	)

	loadDuration, _ := meter.Float64Counter(
		"ollama_load_duration_seconds",
		metric.WithDescription("The request load duration in seconds."),
		metric.WithUnit("seconds"),
	)

	promptEvalCount, _ := meter.Int64Counter(
		"ollama_prompt_eval_total",
		metric.WithDescription("The number of prompt token evaluated."),
		metric.WithUnit("tokens"),
	)

	promptEvalDuration, _ := meter.Float64Counter(
		"ollama_prompt_eval_duration_seconds",
		metric.WithDescription("The prompt evaluation duration in seconds."),
		metric.WithUnit("seconds"),
	)

	evalCount, _ := meter.Int64Counter(
		"ollama_eval_total",
		metric.WithDescription("The number of token evaluated."),
		metric.WithUnit("tokens"),
	)

	evalDuration, _ := meter.Float64Counter(
		"ollama_eval_duration_seconds",
		metric.WithDescription("The prompt evaluation duration in seconds."),
		metric.WithUnit("seconds"),
	)

	peakMemory, _ := meter.Int64Gauge(
		"ollama_peak_memory_bytes",
		metric.WithDescription("The peak memory used during the computation in bytes."),
		metric.WithUnit("bytes"),
	)

	runningModelsCount, _ := meter.Int64Gauge(
		"ollama_running_models",
		metric.WithDescription("Number of currently running models."),
		metric.WithUnit("{models}"),
	)

	modelSizeBytes, _ := meter.Int64Gauge(
		"ollama_model_size_bytes",
		metric.WithDescription("Total size of the loaded model in bytes (from ollama ps)."),
		metric.WithUnit("bytes"),
	)

	modelVRAMBytes, _ := meter.Int64Gauge(
		"ollama_model_vram_bytes",
		metric.WithDescription("GPU VRAM usage of the loaded model in bytes (from ollama ps)."),
		metric.WithUnit("bytes"),
	)

	modelContextLength, _ := meter.Int64Gauge(
		"ollama_model_context_length",
		metric.WithDescription("Context length of the loaded model (from ollama ps)."),
		metric.WithUnit("{tokens}"),
	)

	modelExpiresAt, _ := meter.Int64Gauge(
		"ollama_model_expires_at_unix",
		metric.WithDescription("Unix timestamp when the model expires from memory (from ollama ps)."),
		metric.WithUnit("seconds"),
	)

	modelGPUPercent, _ := meter.Float64Gauge(
		"ollama_model_gpu_percent",
		metric.WithDescription("Percentage of model loaded on GPU (0-100, from ollama ps)."),
		metric.WithUnit("%"),
	)

	return &Metrics{
		Start:              build,
		Requests:           req,
		TotalDuration:      totalDuration,
		LoadDuration:       loadDuration,
		PromptEvalCount:    promptEvalCount,
		PromptEvalDuration: promptEvalDuration,
		EvalCount:          evalCount,
		EvalDuration:       evalDuration,
		PeakMemory:         peakMemory,
		RunningModelsCount: runningModelsCount,
		ModelSizeBytes:     modelSizeBytes,
		ModelVRAMBytes:     modelVRAMBytes,
		ModelContextLength: modelContextLength,
		ModelExpiresAt:     modelExpiresAt,
		ModelGPUPercent:    modelGPUPercent,
	}
}

func (m *Metrics) RecordRequests(ctx context.Context, action string, statusCode int64, status string) {
	m.Requests.Add(ctx, 1, metric.WithAttributes(
		attribute.String("action", action),
		attribute.Int64("status_code", statusCode),
		attribute.String("status", status),
	))
}

// UpdateRunningModels records metrics for all currently running models.
// This should be called whenever the set of running models changes or before exposing metrics.
func (m *Metrics) UpdateRunningModels(ctx context.Context, models []RunningModelInfo) {
	for _, mod := range models {
		attrs := []attribute.KeyValue{
			attribute.String("name", mod.Name),
			attribute.String("digest", mod.Digest),
			attribute.String("format", mod.Format),
			attribute.String("family", mod.Family),
			attribute.String("parameter_size", mod.ParameterSize),
			attribute.String("quantization_level", mod.QuantizationLevel),
		}

		m.ModelSizeBytes.Record(ctx, mod.Size, metric.WithAttributes(attrs...))
		m.ModelVRAMBytes.Record(ctx, mod.SizeVRAM, metric.WithAttributes(attrs...))
		m.ModelContextLength.Record(ctx, int64(mod.ContextLength), metric.WithAttributes(attrs...))
		m.ModelExpiresAt.Record(ctx, mod.ExpiresAt.Unix(), metric.WithAttributes(attrs...))
		m.ModelGPUPercent.Record(ctx, mod.GPUPercent, metric.WithAttributes(attrs...))
	}

	m.RunningModelsCount.Record(ctx, int64(len(models)))
}

func NewPrometheusMeterProvider(res *resource.Resource, exp *prometheus.Exporter) (*sdkmetric.MeterProvider, error) {
	if exp == nil {
		return nil, errors.New("exporter cannot be nil")
	}
	meterProvider := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(exp),
	)

	// Start go runtime metric collection.
	err := runtime.Start(runtime.WithMeterProvider(meterProvider),
		runtime.WithMinimumReadMemStatsInterval(time.Second))
	if err != nil {
		return nil, err
	}

	return meterProvider, nil
}

func InitMetrics() (*Metrics, error) {
	res, err := resource.New(context.Background(),
		resource.WithAttributes(
			semconv.ServiceNameKey.String(namespace),
			semconv.ServiceVersionKey.String("v0.1.0"),
		),
		resource.WithProcessRuntimeDescription(),
	)
	if err != nil {
		return nil, err
	}

	exporter, err := prometheus.New()
	if err != nil {
		return nil, err
	}

	mp, err := NewPrometheusMeterProvider(res, exporter)
	if err != nil {
		return nil, err
	}
	otel.SetMeterProvider(mp)

	meter := mp.Meter(namespace, metric.WithInstrumentationVersion(runtime.Version()))
	return NewMetrics(meter), nil
}
