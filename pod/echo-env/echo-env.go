package main

import (
	"fmt"
	"net/http"
	"os"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
	echo := os.Getenv("ECHO")
	fmt.Fprintln(w, echo)
}

func listenAndServe(port string) {
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		panic("ListenAndServe: " + err.Error())
	}
}

func main() {
	fmt.Println("Starting echo server with echo: ", os.Getenv("ECHO"))
	http.HandleFunc("/", helloHandler)
	port := "8080"
	go listenAndServe(port)

	select {}
}
