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
      (let [scores (map name-score (map #(cstr/replace % "\"" "" ) 
                                    (sort (cstr/split line #","))))
            max-names (count scores)]
          (loop [total-score 0 index 0] 
            (if (>= index max-names) total-score
              (recur (+ total-score (* (inc index) (nth scores index))) (inc index)))
            )
        )
      )      
    )
  )

(defn solve 
  [data-file]
  (load-data data-file)
  )
