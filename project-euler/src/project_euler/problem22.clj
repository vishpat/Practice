(ns project-euler.problem22
 (:require [clojure.java.io :as io]))

(defn load-data
  [data-file]
  (with-open [rdr (io/reader data-file)]
    (doseq [line (line-seq rdr)]
      (println (clojure.string/split line #","))
      )      
    )
  )

(defn solve 
  [data-file]
  (load-data data-file)
  )
