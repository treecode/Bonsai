/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// GLSL shaders for particle rendering
#define STRINGIFY(A) #A

const char *simpleVS = STRINGIFY(
    void main()                                                 \n
    {                                                           \n
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                   \n
    gl_Position = gl_ModelViewProjectionMatrix * wpos;      \n
    gl_FrontColor = gl_Color;                               \n
    }                                                           \n
    );

// particle vertex shader
const char *particleVS = STRINGIFY(
    attribute float pointRadiusAttr;                          \n
    uniform float pointRadius;  // point size in world space    \n
    uniform float pointScale;   // scale to calculate size in pixels \n
    uniform float overBright;
    uniform float overBrightThreshold;
    uniform float ageScale;
    uniform float dustAlpha;
    uniform float fogDist;
    uniform float cullDarkMatter;
    out varying vec4 vpos;                                 \n
    out varying vec4 vcol;                                 \n
    out varying float vsize;                               \n
    void main()                                                 \n
    {                                                           \n
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                   \n
    gl_Position = gl_ModelViewProjectionMatrix * wpos;      \n
    float mass = gl_Vertex.w;
    float type = gl_Color.w;

    float pointRadius1 = pointRadius; \n
//    float pointRadius1 = pointRadiusAttr; \n

    // calculate window-space point size                    \n
    vec4 eyeSpacePos = gl_ModelViewMatrix * wpos;           \n
    float dist = length(eyeSpacePos.xyz);                   \n
    //float dist = -eyeSpacePos.z; \n

    //pointRadius1 *= 1.0 + smoothstep(overBrightThreshold, 0.0, age)*ageScale;
    //pointRadius1 *= mass;
    vec4 col = gl_Color;
    //    type = 3.0;
    if (type == 0.0) {
      // dust
      pointRadius1 *= ageScale;	// scale up
      col.a = dustAlpha;
    } else if (type == 1.0) {
      col.rgb *= overBrightThreshold; // XXX hack, using this var for now!!
    } else if (type == 2.0) {
      // star
      col.rgb *= overBright;
      col.a = 1.0;
    } else if (type == 3.0) {
      //col.a = 1.0;
      // cull dark matter
      if (cullDarkMatter != 0) {
        gl_Position.w = -1.0;
      }
    }

    //gl_PointSize = pointRadius1*(pointScale / dist);       \n
    gl_PointSize = max(1.0, pointRadius1 * (pointScale / dist)); \n
    //gl_PointSize = max(1.0, pointRadius1 * (2*pointScale / dist)); \n
      //float pointSize = pointRadius1 * (pointScale / dist);
      //if (pointSize < 1.0) col.rgb *= pointSize;
      //gl_PointSize = max(1.0, pointSize);

      //gl_TexCoord[0] = vec4(gl_MultiTexCoord0.xyz, age); // sprite texcoord  \n
      //gl_TexCoord[1] = vec4(eyeSpacePos.xyz, mass);                           \n
      gl_TexCoord[1] = vec4(eyeSpacePos.xyz, type);

    float fog = exp(-dist*fogDist);

    //gl_FrontColor = gl_Color;                               \n
    gl_FrontColor = vec4(col.rgb*fog, col.a);       \n
      //gl_FrontColor = vec4(gl_Color.xyz, 1.0);                \n
      vpos   = wpos;  \n
      vcol   = gl_FrontColor;                           \n
      vsize  = gl_PointSize*dist;                       \n
    }                                                           \n
);

// motion blur shaders
const char *mblurVS =
"#version 140\n"
"#extension GL_ARB_compatibility : enable\n"
STRINGIFY(
    uniform samplerBuffer positionSampler;
    uniform float timestep;                                    \n
    uniform vec3 sortVector;
    //uniform float sliceNo;
    //uniform float numSlices;
    //uniform float numParticles;
    uniform float sliceZ;
    uniform float invSliceWidth;
    void main()                                                \n
    {                                                          \n
    vec3 pos    = gl_Vertex.xyz;                           \n
    vec3 vel    = gl_MultiTexCoord0.xyz;                   \n
    vec3 pos2   = (pos - vel*timestep).xyz;                \n // previous position \n

    vec4 eyePos = gl_ModelViewMatrix * vec4(pos, 1.0);  \n // eye space
    gl_Position = eyePos;
    gl_TexCoord[0] = gl_ModelViewMatrix * vec4(pos2, 1.0); \n

    // aging
    float age = gl_Vertex.w;                                 \n
    //float lifetime = gl_MultiTexCoord0.w;                    \n
    //float phase = (lifetime > 0.0) ? (age / lifetime) : 1.0; \n	// [0, 1]

    //	gl_TexCoord[1].x = phase;                                \n
    gl_TexCoord[1].x = age; \n
    gl_TexCoord[1].y = float(gl_VertexID); \n

    // fade out based on age
    //	float fade = 1.0 - phase;                                \n
    //    float fade = 1.0 - smoothstep(1.1, 1.2, age); \n
    float fade = 1.0;

// calc slice position
/*
   vec3 minPos = texelFetchBuffer(positionSampler, 0).xyz; \n
   vec3 maxPos = texelFetchBuffer(positionSampler, int(numParticles)-1).xyz; \n
   float mint = -dot(minPos, sortVector); \n
   float maxt = -dot(maxPos, sortVector); \n
   float t = -dot(pos, sortVector); \n
   float u = (t - mint) / (maxt - mint); // [0, 1]\n
   u = frac(u * numSlices);
   */

// AA calc
//float f = 1.0 - (abs(eyePos.z - sliceZ) * invSliceWidth);
//f = clamp(f, 0.0, 1.0);

//    gl_FrontColor = gl_Color;                              \n
//    gl_FrontColor = vec4(1, 1-age, 0, gl_Color.w);    // yellow->red
gl_FrontColor = vec4(gl_Color.xyz, gl_Color.w*fade);     \n
  //gl_FrontColor = vec4(f, f, f, 1.0);
  //gl_FrontColor = vec4(f, f, f, gl_Color.w*fade);
  //gl_FrontColor = vec4(u.xxx, 1.0); \n
    }                                                            \n
);

// motion blur geometry shader
// - outputs stretched quad between previous and current positions
const char *mblurGS = 
"#version 120\n"
"#extension GL_EXT_gpu_shader4 : enable\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
STRINGIFY(
    attribute float pointRadiusAttr;                          \n
    uniform float pointRadius;  // point size in world space       \n
    uniform float ageScale;
    void main()                                                    \n
    {                                                              \n
    // aging                                                   \n
    float phase = gl_TexCoordIn[0][1].x;                       \n
    float radius = pointRadius;                                \n

    // scale size based on age
    //radius *= 1.0 + smoothstep(1.0, 1.2, phase)*2.0;
    //radius *= 1.0 + smoothstep(0.5, 0.0, phase)*ageScale;

    // eye space                                               \n
    vec3 pos = gl_PositionIn[0].xyz;                           \n
    vec3 pos2 = gl_TexCoordIn[0][0].xyz;                       \n
    vec3 motion = pos - pos2;                                  \n
    vec3 dir = normalize(motion);                              \n
    float len = length(motion);                                \n

    vec3 x = dir * radius;                                     \n
    vec3 view = normalize(-pos);                               \n
    vec3 y = normalize(cross(dir, view)) * radius;             \n
    float facing = dot(view, dir);                             \n

    // check for very small motion to avoid jitter             \n
    float threshold = 0.01;                                    \n
    if ((len < threshold) || (facing > 0.95) || (facing < -0.95)) {  \n
      pos2 = pos;                                            \n
        x = vec3(radius, 0.0, 0.0);                            \n
        y = vec3(0.0, -radius, 0.0);                           \n
    }                                                          \n

//    gl_PrimitiveID = gl_PrimitiveIDIn;
gl_PrimitiveID = int(gl_TexCoordIn[0][1].y);

// output quad                                             \n
gl_FrontColor = gl_FrontColorIn[0];                        \n
  gl_TexCoord[0] = vec4(0, 0, 0, phase);                     \n
  gl_TexCoord[1] = gl_PositionIn[0];                         \n
  gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);  \n
  EmitVertex();                                              \n

  gl_TexCoord[0] = vec4(0, 1, 0, phase);                     \n
  gl_TexCoord[1] = gl_PositionIn[0];                         \n
  gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);  \n
  EmitVertex();                                              \n

  gl_TexCoord[0] = vec4(1, 0, 0, phase);                     \n
  gl_TexCoord[1] = gl_PositionIn[0];                         \n
  gl_Position = gl_ProjectionMatrix * vec4(pos2 - x + y, 1); \n
  EmitVertex();                                              \n

  gl_TexCoord[0] = vec4(1, 1, 0, phase);                     \n
  gl_TexCoord[1] = gl_PositionIn[0];                         \n
  gl_Position = gl_ProjectionMatrix * vec4(pos2 - x - y, 1); \n
  EmitVertex();                                              \n
    }                                                              \n
);


const char *simplePS = STRINGIFY(
    void main()                                                    \n
    {                                                              \n
    gl_FragColor = gl_Color;                                   \n
    }                                                              \n
    );

// render particle without shadows (to shadowmap)
const char *particlePS = 
"#version 120\n"
"#extension GL_EXT_gpu_shader4 : enable\n"
STRINGIFY(
    uniform sampler2D rampTex;
    //uniform sampler2DArray spriteTex;
    uniform sampler2D spriteTex;
    uniform float pointRadius;                                         \n
    uniform float overBright = 1.0;
    uniform float overBrightThreshold;
    uniform float alphaScale;
    uniform float transmission;
    void main()                                                        \n
    {                                                                  \n
    // calculate eye-space sphere normal from texture coordinates  \n
    //vec3 N;                                                        \n
    //N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    //float r2 = dot(N.xy, N.xy);                                    \n
    //if (r2 > 1.0) discard;   // kill pixels outside circle         \n
    //N.z = sqrt(1.0-r2);                                            \n

    //    float alpha = clamp(1.0 - r2, 0.0, 1.0);                     \n
    //    float alpha = exp(-r2*5.0);
    //    float alpha = texture2DArray(spriteTex, vec3(gl_TexCoord[0].xy, gl_PrimitiveID & 7)).x;
      float alpha = texture2D(spriteTex, gl_TexCoord[0].xy).x;		\n
        //alpha *= gl_Color.w;                                           \n
        alpha *= gl_Color.w * alphaScale;
      alpha = clamp(alpha, 0.0, 1.0);

      // color based on age/temp
      //    float mass = gl_TexCoord[0].w;
      float type = gl_TexCoord[1].w;
      //    vec4 col = texture2D(rampTex, vec2(age, 0));
      //	vec4 col = mix(vec4(1.0, 1.0, 0.5, 1), vec4(0.0, 0.0, 0.1, 1), age);	// star color
      //    vec4 col = vec4(0.1);
      //    vec4 col = vec4(age);
      //    col.rgb *= overBright;
      //    col.rgb *= 1.0 + smoothstep(overBrightThreshold, 0.0, age)*overBright;
      //alpha *= smoothstep(1.0, 0.8, age);

      gl_FragColor = vec4(gl_Color.xyz * alpha, max(0, alpha - transmission));              \n
        //    gl_FragColor = vec4(gl_Color.xyz * gl_Color.w, gl_Color.w);              \n
        //    gl_FragColor = vec4(col.xyz * alpha, alpha);
        //    gl_FragColor = vec4(gl_Color.xyz * col.xyz * alpha, alpha); // premul
        //       gl_FragColor = vec4(gl_Color.xyz * col.xyz, alpha);
        //	gl_FragColor = gl_Color;
    }                                                                  \n
);

// render particle including shadows
const char *particleShadowPS = 
"#version 120\n"
"#extension GL_EXT_gpu_shader4 : enable\n"
STRINGIFY(
    uniform sampler2D rampTex;
    //uniform sampler2DArray spriteTex;
    uniform sampler2D spriteTex;
    uniform sampler2D shadowTex;                                       \n
    //uniform sampler2D depthTex;                                        \n
    uniform float pointRadius;                                         \n
    uniform vec2 shadowTexScale;
    uniform float overBright = 1.0;
    uniform float overBrightThreshold;
    uniform float indirectAmount;
    uniform float alphaScale;
    void main()                                                        \n
    {                                                                  \n
    // calculate eye-space sphere normal from texture coordinates  \n
    //vec3 N;                                                        \n
    //N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    //float r2 = dot(N.xy, N.xy);                                    \n
    //if (r2 > 1.0) discard;                                         \n // kill pixels outside circle
    //N.z = sqrt(1.0-r2);                                            \n

    // fetch indirect lighting
      vec4 eyeSpacePos = gl_TexCoord[1];                             \n
        //vec4 eyeSpaceSpherePos = vec4(eyeSpacePos.xyz + N*pointRadius, 1.0); \n // point on sphere
        //vec4 shadowPos = gl_TextureMatrix[0] * eyeSpaceSpherePos;      \n
        vec4 shadowPos = gl_TextureMatrix[0] * eyeSpacePos;      \n
        //    vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n
        //    shadowPos.xy *= shadowTexScale;
        vec3 shadow = texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n

        //float alpha = clamp(1.0 - r2, 0.0, 1.0);                    \n
        //float alpha = exp(-r2*5.0);
        //float alpha = texture2DArray(spriteTex, vec3(gl_TexCoord[0].xy, float(gl_PrimitiveID & 7))).x;
        float alpha = texture2D(spriteTex, gl_TexCoord[0].xy).x;		\n
        //alpha *= gl_Color.w;                                           \n
        alpha *= gl_Color.w * alphaScale;
      alpha = clamp(alpha, 0.0, 1.0);

      // color based on age/temp
      float type = gl_TexCoord[1].w;
      //vec4 col = texture2D(rampTex, vec2(age, 0));
      //vec4 col = mix(vec4(1.0, 1.0, 0.5, 1), vec4(0.0, 0.0, 0.1, 1), age);	// star color
      //col.rgb *= 1.0 + smoothstep(overBrightThreshold, 0.0, age)*overBright;
      //alpha *= smoothstep(1.0, 0.8, age);

      gl_FragColor = vec4(mix(gl_Color.rgb, shadow, indirectAmount)*alpha, alpha);
      //    gl_FragColor = vec4(gl_Color.xyz * shadow * alpha, alpha);     \n // premul alpha
      //    gl_FragColor = vec4(gl_Color.xyz * shadow, alpha);     \n
      //    gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);     \n // premul alpha
      //    gl_FragColor = vec4(gl_Color.xyz, alpha);
      //    gl_FragColor = vec4(shadow * alpha, alpha);     \n // premul alpha
      //	gl_FragColor = vec4(shadow, alpha);     \n
      //	gl_FragColor = gl_Color;
    }
);

// render particle without shadows (to shadowmap)
// with slice anti-aliasing
const char *particleAAPS = 
"#version 120\n"
"#extension GL_EXT_gpu_shader4 : enable\n"
STRINGIFY(
    uniform sampler2D rampTex;
    uniform sampler2DArray spriteTex;
    uniform float pointRadius;                                         \n
    uniform float overBright = 1.0;
    uniform float overBrightThreshold;
    void main()                                                        \n
    {                                                                  \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n
    if (r2 > 1.0) discard;   // kill pixels outside circle         \n
    N.z = sqrt(1.0-r2);                                            \n

    //    float alpha = clamp(1.0 - r2, 0.0, 1.0);                     \n
    float alpha = exp(-r2*4.0);
    //    float alpha = texture2DArray(spriteTex, vec3(gl_TexCoord[0].xy, gl_PrimitiveID & 7)).x;
    alpha *= gl_Color.w;                                           \n

    // color based on age/temp
      float age = gl_TexCoord[0].w;
    vec4 col = texture2D(rampTex, vec2(age, 0));
    //    col.rgb *= overBright;
    col.rgb *= 1.0 + smoothstep(overBrightThreshold, 0.0, age)*overBright;

    //alpha *= smoothstep(1.0, 0.8, age);

    gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);              \n
      //    gl_FragColor = vec4(col.xyz * alpha, alpha);
      //    gl_FragColor = vec4(gl_Color.xyz * col.xyz * alpha, alpha);
      //	gl_FragColor = gl_Color;
    }                                                                  \n
);

// render particle as lit sphere
const char *particleSpherePS = STRINGIFY(
    uniform float pointRadius;                                         \n
    uniform vec3 lightDir = vec3(0.577, 0.577, 0.577);                 \n
    void main()                                                        \n
    {                                                                  \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n
    if (r2 > 1.0) discard;   // kill pixels outside circle         \n
    N.z = sqrt(1.0-r2);                                            \n

    // calculate depth                                             \n
    vec4 eyeSpacePos = vec4(gl_TexCoord[1].xyz + N*pointRadius, 1.0);   // position of this pixel on sphere in eye space \n
    vec4 clipSpacePos = gl_ProjectionMatrix * eyeSpacePos;         \n
    gl_FragDepth = (clipSpacePos.z / clipSpacePos.w)*0.5+0.5;      \n

    float diffuse = max(0.0, dot(N, lightDir));                    \n

    gl_FragColor = diffuse*gl_Color;                               \n
    }                                                                  \n
);

const char *passThruVS = STRINGIFY(
    void main()                                                        \n
    {                                                                  \n
    gl_Position = gl_Vertex;                                       \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
    gl_FrontColor = gl_Color;                                      \n
    }                                                                  \n
    );

const char *transformVS = STRINGIFY(
    void main()                                                        \n
    {                                                                  \n
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;        \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
    gl_FrontColor = gl_Color;                                      \n
    }                                                                  \n
    );

const char *texture2DPS = STRINGIFY(
    uniform sampler2D tex;                                             \n
    uniform float scale;
    uniform float gamma;
    void main()                                                        \n
    {                                                                  \n
    vec4 c = texture2D(tex, gl_TexCoord[0].xy);                    \n
    c.rgb *= scale;
    c.rgb = pow(c.rgb, gamma);                                     \n
    gl_FragColor = c;                                              \n
    }                                                                  \n
    );


// 4 tap 3x3 gaussian blur
const char *blurPS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    uniform float blurRadius;                                                             \n
    void main()                                                                           \n
    {                                                                                     \n
    vec4 c;                                                                           \n
    vec2 uv = gl_TexCoord[0].xy;
    c = texture2D(tex, uv + vec2(-0.5, -0.5)*texelSize*blurRadius);    \n
    c += texture2D(tex, uv + vec2(0.5, -0.5)*texelSize*blurRadius);    \n
    c += texture2D(tex, uv + vec2(0.5, 0.5)*texelSize*blurRadius);     \n
    c += texture2D(tex, uv + vec2(-0.5, 0.5)*texelSize*blurRadius);    \n
    c *= 0.25;                                                                        \n

    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
    );

const char *blur3x3PS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    uniform float blurRadius;                                                             \n
    void main()                                                                           \n
    {                                                                                     \n
    vec4 c;                                                                           \n
    vec2 uv = gl_TexCoord[0].xy;
    c = texture2D(tex, uv + vec2(-1, -1)*texelSize*blurRadius);    \n
    c += texture2D(tex, uv + vec2(0, -1)*texelSize*blurRadius)*2;  \n
    c += texture2D(tex, uv + vec2(1, -1)*texelSize*blurRadius);    \n

    c += texture2D(tex, uv + vec2(-1, 0)*texelSize*blurRadius)*2;  \n
    c += texture2D(tex, uv + vec2(0, 0)*texelSize*blurRadius)*4;   \n
    c += texture2D(tex, uv + vec2(1, 0)*texelSize*blurRadius)*2;   \n

    c += texture2D(tex, uv + vec2(-1, 1)*texelSize*blurRadius);    \n
    c += texture2D(tex, uv + vec2(0, 1)*texelSize*blurRadius)*2;   \n
    c += texture2D(tex, uv + vec2(1, 1)*texelSize*blurRadius);     \n

    c /= 16.0;                                                                        \n

    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
);

const char *blur2PS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    uniform float blurRadius;                                                             \n
    void main()                                                                           \n
    {                                                                                     \n
    vec4 c;                                                                           \n
    vec2 uv = gl_TexCoord[0].xy;
    //c = texture2D(tex, uv - texelSize*blurRadius);    \n
    //c += texture2D(tex, uv)*2;                        \n
    //c += texture2D(tex, uv + texelSize*blurRadius);   \n
    //c *= 0.25;                                                                        \n

    c = texture2D(tex, uv - 2*texelSize*blurRadius);    \n
    c += texture2D(tex, uv - texelSize*blurRadius)*4;   \n
    c += texture2D(tex, uv)*6;                          \n
    c += texture2D(tex, uv + texelSize*blurRadius)*4;   \n
    c += texture2D(tex, uv + 2*texelSize*blurRadius);   \n
    c *= 0.0625;    // / 16                                                           \n

    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
);

const char *thresholdPS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform float threshold;
    uniform float scale;
    void main()                                                                           \n
    {
    vec4 s = texture2D(tex, gl_TexCoord[0].xy);	\n
    float i = dot(s.rgb, vec3(0.333));
    s *= smoothstep(threshold, threshold+0.1, i);
    //s = pow(s, scale);
    s *= scale;
    gl_FragColor = s;
    }
    );

const char *starFilterPS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform sampler2D kernelTex;                                                          \n
    uniform vec2 texelSize;                                                               \n
    uniform float radius;																  \n
    void main()                                                                           \n
    {                                                                                     \n
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);                                                               \n
    float wsum = 0.0;
    vec2 uv = gl_TexCoord[0].xy + texelSize*0.25;
    for(int i=-radius; i<=radius; i++) {                                              \n
    //float x = (i / radius)*3.0;
    //float w = exp(-x*x);
    //float w = 1.0;
    float t = i / radius;
    float w = 1.0 - abs(t);	// triangle filter
    vec4 s = texture2D(tex, uv + i*texelSize);				      \n
    vec4 k = texture2D(kernelTex, vec2(t*2, 0));
    //k.rgb += vec3(0.1);
    c += w * k * s;
    wsum += w;
    }
    //c /= radius*2+1;
    c /= wsum;
    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
);

// downsample image 4 times using 4 bilinear lookups
const char *downSample4PS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    void main()                                                                           \n
    {                                                                                     \n
    vec2 uv = gl_TexCoord[0].xy - texelSize*0.5;
    vec4 c;                                                                           \n
    c = texture2D(tex, uv);				                              \n
    c += texture2D(tex, uv + vec2(texelSize.x*2, 0));				  \n
    c += texture2D(tex, uv + vec2(0, texelSize.y*2));				  \n
    c += texture2D(tex, uv + vec2(texelSize.x*2, texelSize.y*2));	      \n
    c *= 0.25;
    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
    );

// downsample image 2 times using 1 bilinear lookup
const char *downSample2PS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    void main()                                                                           \n
    {                                                                                     \n
    vec2 uv = gl_TexCoord[0].xy;
    vec4 c = texture2D(tex, uv);													  \n
    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
    );

const char *gaussianBlurPS = STRINGIFY(
    uniform sampler2D tex;                                                                \n
    uniform vec2 texelSize;                                                               \n
    uniform float radius;																  \n
    void main()                                                                           \n
    {                                                                                     \n
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);                                                               \n
    float wsum = 0.0;
    //vec2 uv = gl_TexCoord[0].xy - texelSize*0.25;
    vec2 uv = gl_TexCoord[0].xy;
    for(int i=-radius; i<=radius; i++) {                                              \n
    float x = (i / radius)*4.0;
    float w = exp(-x*x);
    vec4 s = texture2D(tex, uv + i*texelSize);				      \n
    c += w * s;
    wsum += w;
    }
    c /= wsum;
    gl_FragColor = c;                                                                 \n
    }                                                                                     \n
    );

const char *compositePS = STRINGIFY(
    uniform sampler2D tex;                                             \n
    uniform sampler2D blurTexH;                                        \n
    uniform sampler2D blurTexV;                                        \n
    uniform sampler2D glowTex;                                         \n
    uniform sampler2D flareTex;                                        \n
    uniform float sourceIntensity;                                     \n
    uniform float glowIntensity;                                       \n
    uniform float starIntensity;
    uniform float flareIntensity;
    uniform float scale = 1.0;                                         \n
    uniform float gamma;
    void main()                                                        \n
    {                                                                  \n
    vec4 c;
    c = texture2D(tex, gl_TexCoord[0].xy) * sourceIntensity;           \n
    if (starIntensity > 0) {
      c += texture2D(blurTexH, gl_TexCoord[0].xy) * starIntensity;
      c += texture2D(blurTexV, gl_TexCoord[0].xy) * starIntensity;
    }
    if (flareIntensity > 0) {
      c += texture2D(flareTex, gl_TexCoord[0].xy) * flareIntensity;
    }
    c += texture2D(glowTex, gl_TexCoord[0].xy) * glowIntensity;
    c.rgb *= scale;

    // vignette
    float d = length(gl_TexCoord[0].xy*2.0-1.0);
    c.rgb *= 1.0 - smoothstep(0.9, 1.5, d);

    c.rgb = pow(c.rgb, gamma);
    gl_FragColor = c;                                              \n
    }                                                                  \n
);

// floor shader
const char *floorVS = STRINGIFY(
    varying vec4 vertexPosEye;  // vertex position in eye space  \n
    varying vec3 normalEye;                                      \n
    void main()                                                  \n
    {                                                            \n
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
    vertexPosEye = gl_ModelViewMatrix * gl_Vertex;           \n
    normalEye = gl_NormalMatrix * gl_Normal;                 \n
    gl_FrontColor = gl_Color;                                \n
    }                                                            \n
    );

const char *floorPS = STRINGIFY(
    uniform vec3 lightPosEye; // light position in eye space                      \n
    uniform vec3 lightColor;                                                      \n
    uniform sampler2D tex;                                                        \n
    uniform sampler2D shadowTex;                                                  \n
    varying vec4 vertexPosEye;  // vertex position in eye space                   \n
    varying vec3 normalEye;                                                       \n
    uniform float brightness = 1.0;
    void main()                                                                   \n
    {                                                                             \n
    vec4 shadowPos = gl_TextureMatrix[0] * vertexPosEye;                      \n
    vec4 colorMap  = texture2D(tex, gl_TexCoord[0].xy);                       \n

    vec3 N = normalize(normalEye);                                            \n
    vec3 L = normalize(lightPosEye - vertexPosEye.xyz);                       \n
    float diffuse = max(0.0, dot(N, L));                                      \n

    //    vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;    \n
    vec3 shadow = texture2DProj(shadowTex, shadowPos.xyw).xyz;
    shadow *= brightness;
    if (shadowPos.w < 0.0) shadow = lightColor;	                              \n // avoid back projections
      gl_FragColor = vec4(gl_Color.xyz * colorMap.xyz * diffuse * shadow, 1.0); \n
    }                                                                             \n
);

const char *volumeVS = STRINGIFY(
    void main()                                                        \n
    {                                                                  \n
    // gl_Vertex already in eye space
    gl_Position = gl_ProjectionMatrix * gl_Vertex;                 \n
    // transform back to world space
    gl_TexCoord[0] = gl_ModelViewMatrixInverse * gl_Vertex;      \n
    //gl_TexCoord[0] = gl_Vertex;
    gl_TexCoord[1] = gl_Vertex;
    gl_FrontColor = gl_Color;                                      \n
    }                                                                  \n
    );

const char *volumePS = STRINGIFY(
    uniform sampler3D noiseTex;                                                           \n
    uniform sampler2D shadowTex;                                       \n
    uniform float noiseFreq;
    uniform float noiseAmp;
    uniform float indirectLighting;
    uniform float volumeStart;
    uniform float volumeWidth;
    float noise(vec3 p)
    {
    return texture3D(noiseTex, p).x;
    }

    float fbm(vec3 p)
    {
    float r;
    r = 0.5000*noise(p); p = p*2.02;
    r += 0.2500*noise(p); p = p*2.03;
    r += 0.1250*noise(p); p = p*2.01;
    r += 0.0625*noise(p);
    //p = p*2.02; r += 0.3125*noise(p);
    return r;
    }

float turb(vec3 p)
{
  float r;
  r = 0.5000*abs(noise(p)); p = p*2.02;
  r += 0.2500*abs(noise(p)); p = p*2.03;
  r += 0.1250*abs(noise(p)); p = p*2.01;
  r += 0.0625*abs(noise(p));
  //p = p*2.02; r += 0.3125*abs(noise(p));
  return r;
}

void main()                                                                           \n
{                                                                                     \n
  vec3 p = gl_TexCoord[0].xyz;
  //float n = abs(texture3D(noiseTex, p*noiseFreq).x);                    \n
  //float n = abs(fbm(p*noiseFreq));
  float n = turb(p*noiseFreq)*noiseAmp;
  float r = length(p);
  //alpha *= smoothstep(0.6, 0.5, r);
  float d = smoothstep(volumeStart + volumeWidth, volumeStart, r + n);
  //float alpha = n * gl_Color.a;
  float alpha = d * gl_Color.a;

  vec4 shadowPos = gl_TextureMatrix[0] * gl_TexCoord[1];      \n
    vec3 shadow = texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n

    //gl_FragColor = vec4(vec3(alpha, alpha, alpha) * alpha, alpha);                                  \n
    //gl_FragColor = vec4(vec3(0, 0, 0) * alpha, alpha);
    //gl_FragColor = vec4(lerp(gl_Color.rgb*d, shadow, indirectLighting) * alpha, alpha);
    gl_FragColor = vec4(lerp(gl_Color.rgb*n, shadow, indirectLighting) * alpha, alpha);
}\n
);

// sky box shader
const char *skyboxVS = STRINGIFY( #version 120\n
    void main()                                                 \n
    {                                                           \n
    //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n
    gl_Position = gl_Vertex; \n
    //gl_TexCoord[0] = gl_MultiTexCoord0;                     \n
    //gl_TexCoord[0] = gl_Vertex;
    //gl_TexCoord[0].xyz = mat3(gl_ModelViewMatrixInverse) * gl_Vertex.xyz; \n
    vec4 eyePos = gl_ProjectionMatrixInverse * gl_Vertex; \n
    eyePos.xyz /= eyePos.w;
    gl_TexCoord[0].xzy = mat3(gl_ModelViewMatrixInverse) * eyePos.xyz;
    gl_FrontColor = gl_Color;                               \n
    }                                                           \n
    );

const char *skyboxPS = STRINGIFY(
    samplerCube tex;
    void main()                                                 \n
    {                                                           \n
    vec4 c = textureCube(tex, gl_TexCoord[0].xyz) * gl_Color; \n
    c.rgb = pow(c.rgb, 2.2);
    gl_FragColor = c;
    //gl_FragColor = textureCube(tex, gl_TexCoord[0].xyz) * gl_Color; \n
    }                                                           \n
    );

const char *splotch2texPS = 
STRINGIFY(
    uniform sampler2D tex;                                             \n
    uniform float scale_pre;                         \n
    uniform float gamma_pre;                         \n
    uniform float scale_post;                         \n
    uniform float gamma_post;
    uniform float sorted; \n
    void main()                                                        \n
    {                                                                  \n
      vec4 c = texture2D(tex, gl_TexCoord[0].xy);                    \n
      c.rgb *= scale_pre;
      c.rgb = pow(c.rgb, gamma_pre);          \n
      if (sorted == 0.0)  \n
      {
        c.rgb = 1.0 - exp(-c.rgb);          \n
        c.rgb *= scale_post;
        c.rgb = pow(c.rgb, gamma_post);          \n
      }  \n
      c.a = 1.0;          \n
      gl_FragColor = c;                                              \n
    }                                                                  \n
  );


const char *splotchVS = 
//"#version 150\n"
STRINGIFY(
    attribute float particleSize;                             \n
    uniform float spriteScale;                                \n
    uniform float starScale;                                  \n
    uniform float starAlpha;                                  \n
    uniform float dmScale;                                    \n
    uniform float dmAlpha;                                    \n
    uniform float spriteSizeMax;                              \n
    uniform float sorted;                                     \n
    out varying vec4 vpos;                                    \n
    out varying vec4 vcol;                                    \n
    out varying float vsize;                                  \n
    void main()                                               \n
    {                                                         \n
      vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                   \n
      float type = gl_Color.w;                                \n
      gl_Position = gl_ModelViewProjectionMatrix * wpos;      \n
                                                              \n
      // calculate window-space point size                    \n
      vec4 eyeSpacePos = gl_ModelViewMatrix * wpos;           \n
      float dist = length(eyeSpacePos.xyz);                   \n
                                                              \n
      // store particle type for PS                           \n
      gl_TexCoord[1] = vec4(0,0,1, type);                     \n
                                                              \n
      float pointSize = particleSize*spriteScale;             \n
      float alpha = 1.0;                                      \n
      vec3 col = gl_Color.rgb;                                \n
      if (type == 0.0)                                        \n
      {                                                       \n
         alpha     *= dmAlpha;                                \n
         pointSize *= dmScale;                                \n
      }                                                       \n
      else                                                    \n
      {                                                       \n
         alpha     *= starAlpha;                              \n
         pointSize *= starScale;                              \n
      }                                                       \n
                                                              \n
      if (sorted == 0.0) col *= 1.0/255;                      \n
      gl_PointSize  = max(spriteSizeMax, pointSize / dist);   \n
      gl_FrontColor = vec4(col, alpha);                       \n
                                                              \n
      vpos   = wpos;                                          \n
      vcol   = gl_FrontColor;                                 \n
      vsize  = gl_PointSize*dist;                             \n
    }                                                         \n
);

const char *splotchGS = 
//"#version 150\n"
STRINGIFY(
    in vec4 vpos[];
    in vec4 vcol[];
    in float vsize[];
    uniform float resx;
    uniform float resy;
    uniform vec4 p0o;
    uniform vec4 p1o;
    uniform vec4 p2o;
    uniform vec4 p3o;
    uniform vec4 p4o;
    uniform vec4 p5o;
    uniform float sorted;                                     
    void main ()
    {
      gl_FrontColor = vcol[0];
      float s = vsize[0];
//      s = min(s, 1024.0f);
      float sx = s / resx;
      float sy = s / resy;

      mat4 invm = transpose(inverse(gl_ModelViewProjectionMatrix));

      vec4 p0 = invm*p0o;
      vec4 p1 = invm*p1o;
      vec4 p2 = invm*p2o;
      vec4 p3 = invm*p3o;
      vec4 p4 = invm*p4o;
      vec4 p5 = invm*p5o;

      vec4 pos = gl_ModelViewProjectionMatrix * vpos[0];

      bool reject = false;
      if (sorted == 0.0)
      {
        gl_ClipDistance[0] = -dot(pos,p0);
        gl_ClipDistance[1] = -dot(pos,p1);
        gl_ClipDistance[2] = -dot(pos,p2);
        gl_ClipDistance[3] = -dot(pos,p3);
        gl_ClipDistance[4] = -dot(pos,p4);
        gl_ClipDistance[5] = -dot(pos,p5);
        for (int k = 0; k < 6; k++)
          if (gl_ClipDistance[k] < 0.0f)
            reject = true;
      }

      if (!reject)
      {
        gl_Position        = pos + vec4(-sx,-sy,0,0);
        gl_TexCoord[0].xy  = vec2(0,0);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(-sx,+sy,0,0);
        gl_TexCoord[0].xy  = vec2(0,1);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(+sx,-sy,0,0);
        gl_TexCoord[0].xy  = vec2(1,0);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(+sx,+sy,0,0);
        gl_TexCoord[0].xy  = vec2(1,1);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();
      }
      
      EndPrimitive();
    }
  );

const char *splotchPS = 
//"#version 150\n"
STRINGIFY(
    uniform float spriteScale;                                \n
    uniform sampler2D spriteTex;                                       \n
    uniform float alphaScale;                                          \n
    uniform float transmission;                                        \n
    uniform float sorted;                                              \n
    void main()                                                        \n
    {                                                                  \n
      float type = gl_TexCoord[1].w;                                   \n
      float alpha = texture2D(spriteTex, gl_TexCoord[0].xy).x;         \n
      vec4 c = vec4(gl_Color.xyz*alpha, 0);
      if (sorted != 0.0)                                               \n
      {                                                                \n
        alpha *= gl_Color.w*alphaScale;                                \n
        alpha = clamp(alpha, 0.0, 1.0);                                \n
        c = vec4(gl_Color.xyz * alpha, max(0, alpha-transmission));    \n
      }                                                                \n
      gl_FragColor = c;                                                \n
    }                                                                  \n
  );

const char *volnewVS = 
//"#version 150\n"
STRINGIFY(
    attribute float particleSize;                             \n
    uniform float spriteScale;                                \n
    uniform float starScale;                                  \n
    uniform float starAlpha;                                  \n
    uniform float dmScale;                                    \n
    uniform float dmAlpha;                                    \n
    uniform float spriteSizeMax;                              \n
    uniform float pointRadius;  // point size in world space    \n
    uniform float overBright;
    uniform float overBrightThreshold;
    uniform float ageScale;                                   \n
    uniform float dustAlpha;
    uniform float fogDist;
    uniform float cullDarkMatter;
    out varying vec4 vpos;                                    \n
    out varying vec4 vcol;                                    \n
    out varying float vsize;                                  \n
    void main()                                               \n
    {                                                         \n
      vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                   \n
      float type = gl_Color.w;                                \n
      gl_Position = gl_ModelViewProjectionMatrix * wpos;      \n
                                                              \n
      // calculate window-space point size                    \n
      vec4 eyeSpacePos = gl_ModelViewMatrix * wpos;           \n
      float dist = length(eyeSpacePos.xyz);                   \n
                                                              \n
      // store particle type for PS                           \n
      gl_TexCoord[1] = vec4(0,0,1, type);                     \n
                                                              \n
      float pointSize = particleSize;                         \n
      vec4 col = gl_Color;                                    \n
      col.a = 1.0;
      if (type == 0.0)                                        \n
      {
        col.a = dustAlpha;
        pointSize = pointRadius * ageScale;
      } 
      else if (type == 1.0) 
      {
        col.rgb *= overBrightThreshold; 
        pointSize = pointRadius;
      }
      else if (type == 2.0) 
      {
        // star
        col.rgb *= overBright;
        pointSize = pointRadius;
      } 
      else if (type == 3.0) 
      {
        if (cullDarkMatter != 0) {
          gl_Position.w = -1.0;
          wpos.w = -1.0;
        }
        pointSize = pointRadius;
      }
      else if (type == 128.0)                                 \n
      {                                                       \n
         col.a     *= dmAlpha;                                \n
         pointSize *= dmScale;                                \n
      }                                                       \n
      else                                                    \n
      {                                                       \n
         col.a     *= starAlpha;                              \n
         pointSize *= starScale;                              \n
      }                                                       \n
      gl_PointSize  = max(spriteSizeMax, pointSize*spriteScale / dist);   \n
      float fog = exp(-dist*fogDist);
      gl_FrontColor = vec4(col.rgb*fog, col.a);                       \n
                                                              \n
      vpos   = wpos;                                          \n
      vcol   = gl_FrontColor;                                 \n
      vsize  = gl_PointSize*dist;                             \n
    }                                                         \n
);

const char *volnewGS = 
//"#version 150\n"
STRINGIFY(
    in vec4 vpos[];
    in vec4 vcol[];
    in float vsize[];
    uniform float resx;
    uniform float resy;
    uniform vec4 p0o;
    uniform vec4 p1o;
    uniform vec4 p2o;
    uniform vec4 p3o;
    uniform vec4 p4o;
    uniform vec4 p5o;
    uniform float sorted;                                     
    void main ()
    {
      gl_FrontColor = vcol[0];
      float s = vsize[0];
      float sx = s / resx;
      float sy = s / resy;

      mat4 invm = transpose(inverse(gl_ModelViewProjectionMatrix));

      vec4 p0 = invm*p0o;
      vec4 p1 = invm*p1o;
      vec4 p2 = invm*p2o;
      vec4 p3 = invm*p3o;
      vec4 p4 = invm*p4o;
      vec4 p5 = invm*p5o;

      vec4 pos = gl_ModelViewProjectionMatrix * vpos[0];

      if (vpos[0] != -1.0)
      {
        gl_Position        = pos + vec4(-sx,-sy,0,0);
        gl_TexCoord[0].xy  = vec2(0,0);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(-sx,+sy,0,0);
        gl_TexCoord[0].xy  = vec2(0,1);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(+sx,-sy,0,0);
        gl_TexCoord[0].xy  = vec2(1,0);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();

        gl_Position        = pos + vec4(+sx,+sy,0,0);
        gl_TexCoord[0].xy  = vec2(1,1);
        gl_ClipDistance[0] = -dot(gl_Position,p0);
        gl_ClipDistance[1] = -dot(gl_Position,p1);
        gl_ClipDistance[2] = -dot(gl_Position,p2);
        gl_ClipDistance[3] = -dot(gl_Position,p3);
        gl_ClipDistance[4] = -dot(gl_Position,p4);
        gl_ClipDistance[5] = -dot(gl_Position,p5);
        EmitVertex();
      }
      
      EndPrimitive();
    }
  );

const char *volnewPS = 
//"#version 150\n"
STRINGIFY(
    uniform float spriteScale;                                \n
    uniform sampler2D spriteTex;                                       \n
    uniform float alphaScale;                                          \n
    uniform float transmission;                                        \n
    void main()                                                        \n
    {                                                                  \n
      float type = gl_TexCoord[1].w;                                   \n
      float alpha = texture2D(spriteTex, gl_TexCoord[0].xy).x;         \n
      alpha *= gl_Color.w*alphaScale;                                \n
      alpha  = clamp(alpha, 0.0, 1.0);                                \n
      gl_FragColor = vec4(gl_Color.xyz * alpha, max(0, alpha-transmission)); \n
    }                                                                  \n
  );

const char *volnew2texPS = 
STRINGIFY(
    uniform sampler2D tex;                                             \n
    uniform float scale;                         \n
    uniform float gamma;
    void main()                                                        \n
    {                                                                  \n
      vec4 c = texture2D(tex, gl_TexCoord[0].xy);                    \n
//      c.rgb *= 100;
      c.rgb = 1.0 - exp(-c.rgb);          \n
      c.rgb *= scale;
      c.rgb = pow(c.rgb, gamma);          \n
      gl_FragColor = c;                                              \n
    }                                                                  \n
  );


