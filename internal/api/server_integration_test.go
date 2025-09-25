package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/LynnColeArt/socialgnn/internal/engine"
)

type httpResponse struct {
	StatusCode int
	Body       []byte
}

func executeRequest(t *testing.T, client *http.Client, method, url string, payload interface{}) httpResponse {
	t.Helper()

	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("failed to marshal payload: %v", err)
		}
		body = bytes.NewReader(data)
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}

	return httpResponse{StatusCode: resp.StatusCode, Body: respBody}
}

func setupIntegrationServer(t *testing.T) (*httptest.Server, *engine.Engine) {
	t.Helper()

	eng := engine.NewEngine()
	eng.LoadSampleData()

	srv := httptest.NewServer(NewServer(eng).Router())
	t.Cleanup(func() {
		srv.Close()
	})

	return srv, eng
}

func TestTrainBatchEndpointValidation(t *testing.T) {
	srv, _ := setupIntegrationServer(t)

	client := srv.Client()

	// Calling train batch without training data should return 400
	res := executeRequest(t, client, http.MethodPost, srv.URL+"/api/train/batch", map[string]interface{}{})
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 when no training data is available, got %d: %s", res.StatusCode, string(res.Body))
	}

	// Add two training examples via API
	payload := map[string]interface{}{
		"user_id": "sarah",
		"item_id": "post1",
		"rating":  1,
	}
	res = executeRequest(t, client, http.MethodPost, srv.URL+"/api/train/example", payload)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 when adding training example, got %d: %s", res.StatusCode, string(res.Body))
	}

	payload["item_id"] = "post2"
	res = executeRequest(t, client, http.MethodPost, srv.URL+"/api/train/example", payload)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 when adding second training example, got %d: %s", res.StatusCode, string(res.Body))
	}

	// Training batch should now succeed even if batch_size exceeds available examples
	res = executeRequest(t, client, http.MethodPost, srv.URL+"/api/train/batch", map[string]interface{}{"epochs": 2, "batch_size": 8})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 when training with available examples, got %d: %s", res.StatusCode, string(res.Body))
	}
}

func TestFollowbackMetricsForNewUser(t *testing.T) {
	srv, _ := setupIntegrationServer(t)
	client := srv.Client()

	// Create a new user via API
	res := executeRequest(t, client, http.MethodPost, srv.URL+"/api/nodes", map[string]interface{}{
		"id":   "newbie",
		"type": "user",
		"metadata": map[string]interface{}{
			"name": "New User",
		},
	})
	if res.StatusCode != http.StatusCreated {
		t.Fatalf("expected 201 when creating user, got %d: %s", res.StatusCode, string(res.Body))
	}

	res = executeRequest(t, client, http.MethodGet, srv.URL+"/api/followback/newbie", nil)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for followback metrics, got %d: %s", res.StatusCode, string(res.Body))
	}

	var metrics struct {
		Following   int     `json:"following"`
		Followers   int     `json:"followers"`
		Followbacks int     `json:"followbacks"`
		HealthScore float64 `json:"health_score"`
	}
	if err := json.Unmarshal(res.Body, &metrics); err != nil {
		t.Fatalf("failed to decode followback metrics: %v", err)
	}

	if metrics.Following != 0 || metrics.Followers != 0 || metrics.Followbacks != 0 {
		t.Fatalf("expected zeroed metrics for new user, got %+v", metrics)
	}
	if metrics.HealthScore == 0 {
		t.Fatalf("expected default health score > 0, got %+v", metrics)
	}
}

func TestRecommendationsEndpoint(t *testing.T) {
	srv, _ := setupIntegrationServer(t)
	client := srv.Client()

	res := executeRequest(t, client, http.MethodGet, srv.URL+"/api/recommendations/sarah?limit=2", nil)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for recommendations, got %d: %s", res.StatusCode, string(res.Body))
	}

	var payload struct {
		Recommendations []struct {
			ID string `json:"id"`
		} `json:"recommendations"`
	}
	if err := json.Unmarshal(res.Body, &payload); err != nil {
		t.Fatalf("failed to decode recommendations: %v", err)
	}

	if len(payload.Recommendations) == 0 {
		t.Fatal("expected at least one recommendation")
	}
}
