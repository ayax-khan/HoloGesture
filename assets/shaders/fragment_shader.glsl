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
    vec3 lightPos = uLightPos;
    vec3 lightColor = vec3(0.0, 0.682, 0.937);

    vec3 ambient = 0.15 * lightColor;

    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - vFragPos);
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
