#include <iostream>
#include <vector>
#include <algorithm>
#include "math/matrix.hpp"

struct Vertex {
    Vec4 pos;
    Vec3 color;
    Vec2 uv;
};


//---Rendering pipeline steps
//--From model space to clip space
void process_vertices(const std::vector<Vertex>& in_vertices, const Mat4& model_mat,
                      const Mat4& view_mat, const Mat4& proj_mat,
                      std::vector<Vertex>& out_vertices) {
    for (const Vertex& vertex : in_vertices) {
        Vertex out_vertex = vertex;
        out_vertex.pos = proj_mat * view_mat * model_mat * out_vertex.pos;
        out_vertices.push_back(out_vertex);
    }
}

//--Vertex post-processing
void clip_against_plane(const std::vector<Vertex>& in_polygon, const Vec4& clipping_plane,
                        std::vector<Vertex>& out_polygon) {
    out_polygon.clear();
    for (size_t idx = 0; idx < in_polygon.size(); idx += 1) {
        Vertex start = in_polygon[idx];
        Vertex end = in_polygon[(idx+1)%in_polygon.size()];
        bool start_in = start.pos.dot(clipping_plane) > 0;
        bool end_in = end.pos.dot(clipping_plane) > 0;
        Vertex intersection; //Intersection occurs when one end is in and the other is out
        if (start_in != end_in) {
            double t = - start.pos.dot(clipping_plane) / (end.pos - start.pos).dot(clipping_plane);
            intersection.pos = start.pos + t * (end.pos - start.pos);
            intersection.color = start.color + t * (end.color - start.color);
            intersection.uv = start.uv + t * (end.uv - start.uv);
        }
        //Add contribution of the current edge start->end to the outpolygon
        if (start_in) {
            out_polygon.push_back(start);
            if (!end_in) out_polygon.push_back(intersection);
        } else if (end_in) out_polygon.push_back(intersection);
    }
}
void post_process_vertices(const std::vector<Vertex>& in_vertices, size_t H, size_t W,
                           std::vector<Vertex>& out_vertices) {
    out_vertices.clear();
    for (size_t triangle_idx = 0; triangle_idx < in_vertices.size()/3; triangle_idx += 1) {
        std::vector<Vertex> subject_polygon {in_vertices[3*triangle_idx],
                                             in_vertices[3*triangle_idx + 1],
                                             in_vertices[3*triangle_idx + 2]};
        //-Back face-culling
        bool back_facing = (subject_polygon[0].pos.pull({0,1,3}).cross(subject_polygon[1].pos.pull({0,1,3}))).dot(subject_polygon[2].pos.pull({0,1,3})) <= 0;
        if (back_facing) continue;

        //-Clipping using Sutherland-Hodgman Algorithm
        std::vector<Vertex> subject_polygon_;
        Vec4 clipping_plane({1.0, 0.0, 0.0, 1.0});
        clip_against_plane(subject_polygon, clipping_plane, subject_polygon_);
        clipping_plane = Vec4({-1.0, 0.0, 0.0, 1.0});
        clip_against_plane(subject_polygon_, clipping_plane, subject_polygon);
        clipping_plane = Vec4({0.0, 1.0, 0.0, 1.0});
        clip_against_plane(subject_polygon, clipping_plane, subject_polygon_);
        clipping_plane = Vec4({0.0, -1.0, 0.0, 1.0});
        clip_against_plane(subject_polygon_, clipping_plane, subject_polygon);
        clipping_plane = Vec4({0.0, 0.0, 1.0, 1.0});
        clip_against_plane(subject_polygon, clipping_plane, subject_polygon_);
        clipping_plane = Vec4({0.0, 0.0, -1.0, 1.0});
        clip_against_plane(subject_polygon_, clipping_plane, subject_polygon);

        //-From clipping space to window space
        for (Vertex& vertex : subject_polygon) {
            //Perspective division (from clipping space to NDC)
            vertex.pos(0) /= vertex.pos(3); //In [-1, 1]
            vertex.pos(1) /= vertex.pos(3); //In [-1, 1]
            vertex.pos(2) /= vertex.pos(3); //In [-1, 1]
            //The last component stores 1/w_c = -1/z_e i.e. the inverse of the positive depth to the camera (needed for perspective correct interpolation)
            vertex.pos(3) = 1.0/vertex.pos(3);
            //From NDC to window space
            vertex.pos(0) = 0.5*W*(vertex.pos(0) + 1.0);
            vertex.pos(1) = 0.5*H*(vertex.pos(1) + 1.0);
            vertex.pos(2) = 0.5*(vertex.pos(2) + 1.0);
        }

        //-Convert polygon to triangles fan
        for (size_t idx = 1; idx < subject_polygon.size() - 1; idx += 1) {
            out_vertices.push_back(subject_polygon[0]);
            out_vertices.push_back(subject_polygon[idx]);
            out_vertices.push_back(subject_polygon[idx+1]);
        }

        //std::cout << triangle_idx << " size polygon " << subject_polygon.size() << std::endl;
    }
}

//-Resterization
void rasterization(const std::vector<Vertex>& in_vertices, size_t H, size_t W,
                   std::vector<Vertex>& out_vertices) {
    out_vertices.clear();
    for (size_t triangle_idx = 0; triangle_idx < in_vertices.size()/3; triangle_idx += 1) {
        Vertex triangle[] {in_vertices[3*triangle_idx],
                           in_vertices[3*triangle_idx + 1],
                           in_vertices[3*triangle_idx + 2]};
        //Compute AABB
        Vec2 min_corner({std::min({floor(triangle[0].pos(0)), floor(triangle[1].pos(0)), floor(triangle[2].pos(0))}),
                         std::min({floor(triangle[0].pos(1)), floor(triangle[1].pos(1)), floor(triangle[2].pos(1))})});
        Vec2 max_corner({std::max({floor(triangle[0].pos(0)), floor(triangle[1].pos(0)), floor(triangle[2].pos(0))}),
                         std::max({floor(triangle[0].pos(1)), floor(triangle[1].pos(1)), floor(triangle[2].pos(1))})});
        //Test for every pixel in AABB if belongs to the triangle (clip to the window in theory no need thanks to clipping)
        for (size_t x = std::max(size_t(min_corner(0)), (size_t)0); x <= std::min(size_t(max_corner(0)), W); x += 1) {
            for (size_t y = std::max(size_t(min_corner(1)), (size_t)0); y <= std::min(size_t(max_corner(1)), H); y += 1) {
                Vec2 pixel({(double)x, (double)y});
                pixel(0) += 0.5;
                pixel(1) += 0.5;
                //Test if pixel in triangle, by computing the "barycentric" coordinate of pixel, becomes barycentric after division by 2*area of triangle
                Vec3 weights;
                for (size_t idx = 0; idx < 3; idx += 1) {
                    Vec2 edge = (triangle[(idx+1)%3].pos - triangle[idx].pos).pull({0, 1});
                    Vec2 pixel_ = pixel - triangle[idx].pos.pull({0, 1});
                    weights((idx+2)%3) = det(edge, pixel_); //Weights given by area parallelogram i.e. det
                }
                bool inside = (weights(0) >= 0) && (weights(1) >= 0) && (weights(2) >= 0);
                if (!inside) continue;
                
                //Perspective correct interpolation of the attributs of the vertices to the pixel
                weights /= det((triangle[1].pos - triangle[0].pos).pull({0, 1}),
                               (triangle[2].pos - triangle[0].pos).pull({0, 1}));
                double fragment_w = weights(0)*triangle[0].pos(3)
                                    + weights(1)*triangle[1].pos(3)
                                    + weights(2)*triangle[2].pos(3); //Inv depth to the camera can be interpolated with the weights computed in window coordinates    
                Vertex fragment;
                fragment.pos = (weights(0)*triangle[0].pos(3)*triangle[0].pos
                              + weights(1)*triangle[1].pos(3)*triangle[1].pos
                              + weights(2)*triangle[2].pos(3)*triangle[2].pos)/fragment_w;
                fragment.color = (weights(0)*triangle[0].pos(3)*triangle[0].color
                                + weights(1)*triangle[1].pos(3)*triangle[1].color
                                + weights(2)*triangle[2].pos(3)*triangle[2].color)/fragment_w;
                fragment.uv = (weights(0)*triangle[0].pos(3)*triangle[0].uv
                             + weights(1)*triangle[1].pos(3)*triangle[1].uv
                             + weights(2)*triangle[2].pos(3)*triangle[2].uv)/fragment_w;
                out_vertices.push_back(fragment);
            }
        }
    }
}

//-Fragment shader and sample processing
//TODO


//---Rendering pipeline
int main(int argc, char *argv[]) {
    //---Scene description
    //-Camera
    Mat4 view_mat = look_at_matrix(Vec3({-1.0, 1.0, 0.0}), Vec3({2.0, 0.0, 0.0}), Vec3({0.0, 1.0, 0.0}));
    Mat4 proj_mat = projection_matrix(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0);
    //-Meshes: two plan, one lying done, the other vertically back to the camera
    std::vector<Vertex> plan {{Vec4({-1.0, 0.0, -1.0, 1.0}), Vec3({1.0, 0.0, 0.0}), Vec2({0.0, 0.0})},
                              {Vec4({-1.0, 0.0, 1.0, 1.0}), Vec3({1.0, 0.0, 0.0}), Vec2({1.0, 0.0})},
                              {Vec4({1.0, 0.0, 1.0, 1.0}), Vec3({0.0, 0.0, 1.0}), Vec2({1.0, 1.0})},

                              {Vec4({-1.0, 0.0, -1.0, 1.0}), Vec3({1.0, 0.0, 0.0}), Vec2({0.0, 0.0})},
                              {Vec4({1.0, 0.0, 1.0, 1.0}), Vec3({0.0, 0.0, 1.0}), Vec2({1.0, 1.0})},
                              {Vec4({1.0, 0.0, -1.0, 1.0}), Vec3({0.0, 0.0, 1.0}), Vec2({0.0, 1.0})}};
    double scale = 1.0; //1.0 no clipping VS 10.0 clipping
    Mat4 model_mat = translation_matrix(Vec3({0.5*scale + 1.0, 0.0, 0.0})) * rotation_matrix(Vec3({0.0, 0.0, 1.0}), 0.0) * scale_matrix(scale);
    Mat4 model_mat_ = translation_matrix(Vec3({0.5*scale + 1.0, 0.0, 0.5*scale})) * rotation_matrix(Vec3({0.0, 0.0, 1.0}), -0.5*M_PI) * scale_matrix(scale);
    //-Window
    size_t H = 100;
    size_t W = 100;

    std::vector<Vertex> buffer_A;
    std::vector<Vertex> buffer_B;
    //---Vertex processing (from model space to clip space)
    process_vertices(plan, model_mat, view_mat, proj_mat, buffer_B);
    process_vertices(plan, model_mat_, view_mat, proj_mat, buffer_B);
    //---Vertex post-processing (back face culling and clipping)
    post_process_vertices(buffer_B, H, W, buffer_A);
    //---Resterization
    rasterization(buffer_A, H, W, buffer_B);
    //---Fragment shader and sample processing
    //TODO


    // for (const Vertex& vet : buffer_B) {
    //     std::cout << vet.pos << std::endl;
    // }
    std::cout << buffer_B.size() << std::endl;
    
}