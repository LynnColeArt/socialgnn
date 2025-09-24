package main

import (
	"log"
	"net/http"
	"os"

	"github.com/LynnColeArt/socialgnn/internal/api"
	"github.com/LynnColeArt/socialgnn/internal/engine"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize the GNN engine
	gnnEngine := engine.NewEngine()

	// Load sample data for demo/testing
	if os.Getenv("LOAD_SAMPLE_DATA") == "true" {
		log.Println("Loading sample Riverside community data...")
		gnnEngine.LoadSampleData()
	}

	// Initialize the API server
	apiServer := api.NewServer(gnnEngine)

	log.Printf("🧠 SocialGNN API Server starting on port %s", port)
	log.Printf("📊 Graph stats: %+v", gnnEngine.GetStats())
	log.Printf("🔗 Endpoints:")
	log.Printf("  POST /api/nodes - Add node")
	log.Printf("  POST /api/edges - Add edge")
	log.Printf("  GET  /api/recommendations/:userID - Get recommendations")
	log.Printf("  GET  /api/similarity/:user1/:user2 - Get user similarity")
	log.Printf("  GET  /api/stats - Get graph statistics")
	log.Printf("  GET  /health - Health check")

	if err := http.ListenAndServe(":"+port, apiServer.Router()); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}