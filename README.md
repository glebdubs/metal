# 3D Rendered Metal Boid Simulation  
  
This project used Apple's Metal API to simulate boid movement in 3D.   
  
To build on your own machine, you may need to add `MetalKit.framework`, `Metal.framework` and `Foundation.framework` to the *Link Binary with Libraries*  
section if building with Xcode.  
You'll also need to download the `metal-cpp` API folder and throw it in the same directory as this project.  
  
Initial configuration: constructs 1024 boids, each represented by a tetrahedron floating in space. They are given a 40x40 unit space to traverse, each   
boid 0.5 units across. The boid's speed is represented by their colour, where a brighter yellow corresponds with a greater velocity.  
    
Boid movement is determined by three factors:  
- If one boid spots another that is `size_t boidInnerRadSq` squared units of distance away from it, or fewer, then they will both be repelled away from each other with a vector of strength `float avoidanceVectorStrength`.
- If one boids spots another that is between `size_t boidInnerRadSq` and `size_t boidOuterRadSq` squared units away from it:
  - A direction convergence vector will be added of strength `float directionConvergenceStrength`, where both boids try to fly in a similar direction to each other.
  - A position convergence vector will be added of strength `float clusteringStrength`, where each boid tries to fly towards the centre mass of all other nearby boids.
  
All of these parameters are variable at the top of `main.mm`  
  
At the moment, the boids use a lazy O(n^2) algorithm to determine their distance from each other, allowing for up to around 1000 - 1500 to run reliably at ~60 FPS  
on my M1 Macbook Air, with only a single CPU core allocated to the simulation. This results in ~60% CPU utilisation (60% of one CPU core, of 8) and 0.2% GPU utilisation.  
  
In future, I'll probably convert the movement algorithm to use an octree or some form of space segmentation to speed up boid movement calculation, as it is extremely CPU intensive in its current state.  
  
To avoid data races in frame read/writing, the simulation also uses a semaphore for frame management, where the `Renderer::draw()` command writes to frame at index `i`,  
while the command buffer reads from index `i-1` in the circular frame buffer `pInstanceDataBuffer`.
