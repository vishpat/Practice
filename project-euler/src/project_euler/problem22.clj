(ns project-euler.problem22
 (:require [clojure.java.io :as io])
 (:require [clojure.string :as cstr]))

(defn name-score
  [n]
  (reduce + (map #(inc (- (int %) (int \A))) n))
  )


(defn load-data
  [data-file]
  (with-open [rdr (io/reader data-file)]
    (doseq [line (line-seq rdr)]
      (println (map name-score (map #(cstr/replace % "\"" "" ) 
                                    (sort (cstr/split line #",")))))
      )      
    )
  )

(defn solve 
  [data-file]
  (load-data data-file)
  )
