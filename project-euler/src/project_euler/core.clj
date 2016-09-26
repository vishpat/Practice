(ns project-euler.core)

(defn sub-1-str
  [m n]
  ()
  )

(defn digit-at
  [number index]
  (Integer/parseInt (str (.charAt number (- (count number) (+ index 1))))))

;(defn add-str
;    [m n]
;  (loop [num1 m num2 n index 0 carry 0 result ""]
;    (let [d1  d2 ] )
;    )
;)
;
;
;(defn multiply-str
;  [m n]
;  (loop [num1 m num2 n result ""]
;    (cond (= num1 "0") "0" (= num2 "0") "0" (= num1 "1") (add-str result num2) 
;          :else (recur (sub-str num1 "1") num2 (add-str result num2)))  
;  )
;)
;
;(defn factorial
;  [n]
;  (
;   loop [x n result 1] 
;   (if (= x 1) result ( recur (- x 1) (* x result))
;   )
;  )
;)
;(defn -main
;  "I don't do a whole lot ... yet."
;  [& args]
;  (println (multiply-str 10 9)))
