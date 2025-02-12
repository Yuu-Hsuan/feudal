Manager Loss (經理損失)
cos
⁡
(
𝜃
)
=
𝑆
DIFF
⋅
goal
∥
𝑆
DIFF
∥
⋅
∥
goal
∥
cos(θ)= 
∥S 
DIFF
​
 ∥⋅∥goal∥
S 
DIFF
​
 ⋅goal
​
 
manager_loss
=
−
𝐸
[
ADV
𝑀
⋅
cos
⁡
(
𝜃
)
]
manager_loss=−E[ADV 
M
​
 ⋅cos(θ)]
說明：

目標 (goal) 和 狀態變化 (S_DIFF) 之間的餘弦相似度衡量兩者方向的一致性。
ADV_M 代表管理層的優勢值。
該公式鼓勵 goal 朝向 S_DIFF 的方向，使得經理的指引能夠促使工人達成目標。
