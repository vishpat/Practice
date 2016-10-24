(ns project-euler.problem22
 (:require [clojure.java.io :as io])
 (:require [clojure.string :as cstr]))

(defn name-score
  [n]
  (reduce + (map #(inc (- (int %) (int \A))) n))
  )

(defn load-and-solve
  [data-file]
  (with-open [rdr (io/reader data-file)]
    (doseq [line (line-seq rdr)]
      (let [scores (map name-score (map #(cstr/replace % "\"" "" ) 
                                    (sort (cstr/split line #","))))]
        (println (reduce + (map-indexed #(* (inc %1) %2) scores)))
       )
      )      
    )
  )

(defn solve 
  [data-file]
  (load-and-solve data-file)
  )
