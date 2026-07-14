VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProjection * uView * worldPos;
}
"""

SOLID_FRAGMENT_SRC = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uColor;
uniform float uGlowIntensity;
uniform float uTime;

out vec4 FragColor;

void main() {
    vec3 lightColor = vec3(0.0, 0.682, 0.937);
    vec3 ambient = 0.15 * lightColor;
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 viewDir = normalize(uViewPos - vFragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;
    float pulse = 0.5 + 0.5 * sin(uTime * 1.5);
    float glow = uGlowIntensity * (0.8 + 0.2 * pulse);
    vec3 result = (ambient + diffuse + specular) * uColor;
    result += glow * vec3(0.0, 0.5, 1.0) * (1.0 - diff);
    float edge = pow(1.0 - max(dot(viewDir, norm), 0.0), 2.0);
    result += edge * glow * 0.3 * vec3(0.0, 0.8, 1.0);
    FragColor = vec4(result, 1.0);
}
"""

WIREFRAME_FRAGMENT_SRC = """
#version 330 core
uniform vec3 uColor;
uniform float uGlowIntensity;
uniform float uTime;

out vec4 FragColor;

void main() {
    float pulse = 0.7 + 0.3 * sin(uTime * 2.0);
    vec3 color = uColor * pulse * (1.0 + uGlowIntensity * 0.5);
    FragColor = vec4(color, 1.0);
}
"""

HOLOGRAPHIC_FRAGMENT_SRC = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uColor;
uniform float uGlowIntensity;
uniform float uTime;

out vec4 FragColor;

void main() {
    vec3 lightColor = vec3(0.0, 0.682, 0.937);
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(uViewPos - vFragPos);
    float fresnel = pow(1.0 - max(dot(viewDir, norm), 0.0), 3.0);

    float scan = 0.5 + 0.5 * sin(vFragPos.y * 8.0 + uTime * 3.0);
    float pulse = 0.6 + 0.4 * sin(uTime * 1.2);

    vec3 color = uColor * (0.3 + 0.7 * diff);
    color += fresnel * uGlowIntensity * vec3(0.0, 0.8, 1.0) * 1.5;
    color += scan * 0.15 * vec3(0.0, 0.5, 1.0);
    color *= pulse;

    float alpha = 0.35 + 0.25 * diff + 0.4 * fresnel;
    FragColor = vec4(color, alpha);
}
"""


SHADER_SOURCES = {
    "solid": (VERTEX_SHADER_SRC, SOLID_FRAGMENT_SRC),
    "wireframe": (VERTEX_SHADER_SRC, WIREFRAME_FRAGMENT_SRC),
    "holographic": (VERTEX_SHADER_SRC, HOLOGRAPHIC_FRAGMENT_SRC),
}
