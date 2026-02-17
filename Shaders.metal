//
//  Shaders.metal
//  05-perspective
//
//  Created by Gleb Dubinin on 27/1/2026.
//  Copyright © 2026 Apple. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct v2f
{
    float4 position [[position]];
    float3 worldPosition;
    half3 color;
};

struct VertexData
{
    float3 position;
};

struct InstanceData
{
    float4x4 instanceTransform;
    float4 instanceColor;
};

struct CameraData
{
    float4x4 perspectiveTransform;
    float4x4 worldTransform;
};

v2f vertex vertexMain( device const VertexData* vertexData [[buffer(0)]],
                       device const InstanceData* instanceData [[buffer(1)]],
                       device const CameraData& cameraData [[buffer(2)]],
                       uint vertexId [[vertex_id]],
                       uint instanceId [[instance_id]] )
{
    v2f o;
    float4 pos = float4( vertexData[ vertexId ].position, 1.0 );
    
    float4 worldPos = instanceData[ instanceId ].instanceTransform * pos;
    o.worldPosition = worldPos.xyz;
    
    pos = instanceData[ instanceId ].instanceTransform * pos;
    pos = cameraData.perspectiveTransform * cameraData.worldTransform * pos;
    o.position = pos;
    o.color = half3( instanceData[ instanceId ].instanceColor.rgb );
    return o;
}

half4 fragment fragmentMain( v2f in [[stage_in]] )
{
    float3 normal = normalize(cross(dfdx(in.worldPosition), dfdy(in.worldPosition)));
    float3 lightDirection = normalize(float3(0.5, -0.5, -1.0));
    float diffuse = saturate(dot(normal, lightDirection));
    float ambient = 0.2;
    float brightness = saturate(diffuse + ambient);
    
    return half4( in.color * brightness, 1.0 );
}
