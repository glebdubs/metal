/*
 *
 * Copyright 2022 Apple Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

//#ifdef x
//    #warning "x was defined as a macro — undefining to avoid metal-cpp clash"
//    #undef x
//#endif
//
//// Also check common case variants
//#ifdef X
//    #undef X
//#endif

#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>

#include <simd/simd.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

static constexpr size_t kNumInstances = 32;
static constexpr size_t lineWidth = 32;
static constexpr size_t kMaxFramesInFlight = 3;
static constexpr size_t objCount = lineWidth * kNumInstances;

static constexpr size_t boidInnerRadSq = 1;
static constexpr size_t boidOuterRadSq = 5;

static constexpr float avoidanceVectorStrength = 8;
static constexpr float directionConvergenceStrength = 2;
static constexpr float clusteringStrength = 0.5;
static constexpr int windowSize = 20;


#pragma region Declarations {

namespace math
{
    constexpr simd::float3 add( const simd::float3& a, const simd::float3& b );
    constexpr simd::float3 sub( const simd::float3& a, const simd::float3& b );
    constexpr simd_float4x4 makeIdentity();
    float randomFloat();
    float getDistanceSquared(simd::float3& a, simd::float3& b);
    float getMagnitude(const simd::float3& a);
    float fclamp(const float a, const float max, const float min);
    simd::float4 f3tf4(const simd::float3& a);
    simd::float3 multByConstant(const simd::float3& a, float b);
    simd::float3 randomFloat3();
    simd::float4x4 makePerspective();
    simd::float4x4 makeXRotate( float angleRadians );
    simd::float4x4 makeYRotate( float angleRadians );
    simd::float4x4 makeZRotate( float angleRadians );
    simd::float4x4 makeTranslate( const simd::float3& v );
    simd::float4x4 makeScale( const simd::float3& v );
}

class Renderer
{
    public:
        Renderer( MTL::Device* pDevice );
        ~Renderer();
        void buildShaders();
        void buildDepthStencilStates();
        void buildBuffers();
        void draw( MTK::View* pView );

    private:
        MTL::Device* _pDevice;
        MTL::CommandQueue* _pCommandQueue;
        MTL::Library* _pShaderLibrary;
        MTL::RenderPipelineState* _pPSO;
        MTL::DepthStencilState* _pDepthStencilState;
        MTL::Buffer* _pVertexDataBuffer;
        MTL::Buffer* _pInstanceDataBuffer[kMaxFramesInFlight];
        MTL::Buffer* _pCameraDataBuffer[kMaxFramesInFlight];
        MTL::Buffer* _pIndexBuffer;
        simd::float3 directions[kMaxFramesInFlight][objCount];
        float _angle;
        int _frame;
        dispatch_semaphore_t _semaphore;
        static const int kMaxFramesInFlight;
};

class MyMTKViewDelegate : public MTK::ViewDelegate
{
    public:
        MyMTKViewDelegate( MTL::Device* pDevice );
        virtual ~MyMTKViewDelegate() override;
        virtual void drawInMTKView( MTK::View* pView ) override;

    private:
        Renderer* _pRenderer;
};

class MyAppDelegate : public NS::ApplicationDelegate
{
    public:
        ~MyAppDelegate();

        NS::Menu* createMenuBar();

        virtual void applicationWillFinishLaunching( NS::Notification* pNotification ) override;
        virtual void applicationDidFinishLaunching( NS::Notification* pNotification ) override;
        virtual bool applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender ) override;

    private:
        NS::Window* _pWindow;
        MTK::View* _pMtkView;
        MTL::Device* _pDevice;
        MyMTKViewDelegate* _pViewDelegate = nullptr;
};

#pragma endregion Declarations }


int main( int argc, char* argv[] )
{
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MyAppDelegate del;

    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate( &del );
    pSharedApplication->run();

    pAutoreleasePool->release();

    return 0;
}


#pragma mark - AppDelegate
#pragma region AppDelegate {

MyAppDelegate::~MyAppDelegate()
{
    _pMtkView->release();
    _pWindow->release();
    _pDevice->release();
    delete _pViewDelegate;
}

NS::Menu* MyAppDelegate::createMenuBar()
{
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Menu* pMainMenu = NS::Menu::alloc()->init();
    NS::MenuItem* pAppMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pAppMenu = NS::Menu::alloc()->init( NS::String::string( "Appname", UTF8StringEncoding ) );

    NS::String* appName = NS::RunningApplication::currentApplication()->localizedName();
    NS::String* quitItemName = NS::String::string( "Quit ", UTF8StringEncoding )->stringByAppendingString( appName );
    SEL quitCb = NS::MenuItem::registerActionCallback( "appQuit", [](void*,SEL,const NS::Object* pSender){
        auto pApp = NS::Application::sharedApplication();
        pApp->terminate( pSender );
    } );

    NS::MenuItem* pAppQuitItem = pAppMenu->addItem( quitItemName, quitCb, NS::String::string( "q", UTF8StringEncoding ) );
    pAppQuitItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );
    pAppMenuItem->setSubmenu( pAppMenu );

    NS::MenuItem* pWindowMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pWindowMenu = NS::Menu::alloc()->init( NS::String::string( "Window", UTF8StringEncoding ) );

    SEL closeWindowCb = NS::MenuItem::registerActionCallback( "windowClose", [](void*, SEL, const NS::Object*){
        auto pApp = NS::Application::sharedApplication();
            pApp->windows()->object< NS::Window >(0)->close();
    } );
    NS::MenuItem* pCloseWindowItem = pWindowMenu->addItem( NS::String::string( "Close Window", UTF8StringEncoding ), closeWindowCb, NS::String::string( "w", UTF8StringEncoding ) );
    pCloseWindowItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );

    pWindowMenuItem->setSubmenu( pWindowMenu );

    pMainMenu->addItem( pAppMenuItem );
    pMainMenu->addItem( pWindowMenuItem );

    pAppMenuItem->release();
    pWindowMenuItem->release();
    pAppMenu->release();
    pWindowMenu->release();

    return pMainMenu->autorelease();
}

void MyAppDelegate::applicationWillFinishLaunching( NS::Notification* pNotification )
{
    NS::Menu* pMenu = createMenuBar();
    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->setMainMenu( pMenu );
    pApp->setActivationPolicy( NS::ActivationPolicy::ActivationPolicyRegular );
}

void MyAppDelegate::applicationDidFinishLaunching( NS::Notification* pNotification )
{
    CGRect frame = (CGRect){ {100.0, 100.0}, {1024, 1024} };

    _pWindow = NS::Window::alloc()->init(
        frame,
        NS::WindowStyleMaskClosable|NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false );

    _pDevice = MTL::CreateSystemDefaultDevice();

    _pMtkView = MTK::View::alloc()->init( frame, _pDevice );
    _pMtkView->setColorPixelFormat( MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );
    _pMtkView->setClearColor( MTL::ClearColor::Make( 0.1, 0.1, 0.1, 1.0 ) );
    _pMtkView->setDepthStencilPixelFormat( MTL::PixelFormat::PixelFormatDepth16Unorm );
    _pMtkView->setClearDepth( 1.0f );

    _pViewDelegate = new MyMTKViewDelegate( _pDevice );
    _pMtkView->setDelegate( _pViewDelegate );

    _pWindow->setContentView( _pMtkView );
    _pWindow->setTitle( NS::String::string( "05 - Perspective", NS::StringEncoding::UTF8StringEncoding ) );

    _pWindow->makeKeyAndOrderFront( nullptr );

    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->activateIgnoringOtherApps( true );
}

bool MyAppDelegate::applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender )
{
    return true;
}

#pragma endregion AppDelegate }


#pragma mark - ViewDelegate
#pragma region ViewDelegate {

MyMTKViewDelegate::MyMTKViewDelegate( MTL::Device* pDevice )
: MTK::ViewDelegate()
, _pRenderer( new Renderer( pDevice ) )
{
}

MyMTKViewDelegate::~MyMTKViewDelegate()
{
    delete _pRenderer;
}

void MyMTKViewDelegate::drawInMTKView( MTK::View* pView )
{
    _pRenderer->draw( pView );
}

#pragma endregion ViewDelegate }


#pragma mark - Math

namespace math
{

    float randomFloat() {
        return (float)rand();
    }

    simd::float3 randomFloat3() {
        return simd::float3{randomFloat(), randomFloat(), randomFloat()};
    }

    float getDistanceSquared(simd::float3& a, simd::float3& b) {
        float x = a.x - b.x;
        float y = a.y - b.y;
        float z = a.z - b.z;
        return (float)( x*x + y*y + z*z );
    }

    constexpr simd::float3 add( const simd::float3& a, const simd::float3& b )
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    constexpr simd::float3 sub( const simd::float3& a, const simd::float3& b )
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    float getMagnitude(const simd::float3& a) {
        return (float)(a.x*a.x + a.y*a.y + a.z*a.z);
    }

    simd::float3 multByConstant( const simd::float3& a, float b) {
        return simd::float3 {a.x*b, a.y*b, a.z*b};
    }

    float fclamp(const float a, const float min, const float max) {
        if(a < min) return min;
        if(a > max) return max;
        return a;
    }

    simd::float4 f3tf4(const simd::float3& a) {
        return simd::float4 {a.x, a.y, a.z, 0};
    }

    constexpr simd_float4x4 makeIdentity()
    {
        using simd::float4;
        return (simd_float4x4){ (float4){ 1.f, 0.f, 0.f, 0.f },
                                (float4){ 0.f, 1.f, 0.f, 0.f },
                                (float4){ 0.f, 0.f, 1.f, 0.f },
                                (float4){ 0.f, 0.f, 0.f, 1.f } };
    }

    simd::float4x4 makePerspective( float fovRadians, float aspect, float znear, float zfar )
    {
        using simd::float4;
        float ys = 1.f / tanf(fovRadians * 0.5f);
        float xs = ys / aspect;
        float zs = zfar / ( znear - zfar );
        return simd_matrix_from_rows((float4){ xs, 0.0f, 0.0f, 0.0f },
                                     (float4){ 0.0f, ys, 0.0f, 0.0f },
                                     (float4){ 0.0f, 0.0f, zs, znear * zs },
                                     (float4){ 0, 0, -1, 0 });
    }

    simd::float4x4 makeXRotate( float angleRadians )
    {
        using simd::float4;
        const float a = angleRadians;
        return simd_matrix_from_rows((float4){ 1.0f, 0.0f, 0.0f, 0.0f },
                                     (float4){ 0.0f, cosf( a ), sinf( a ), 0.0f },
                                     (float4){ 0.0f, -sinf( a ), cosf( a ), 0.0f },
                                     (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
    }

    simd::float4x4 makeYRotate( float angleRadians )
    {
        using simd::float4;
        const float a = angleRadians;
        return simd_matrix_from_rows((float4){ cosf( a ), 0.0f, sinf( a ), 0.0f },
                                     (float4){ 0.0f, 1.0f, 0.0f, 0.0f },
                                     (float4){ -sinf( a ), 0.0f, cosf( a ), 0.0f },
                                     (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
    }

    simd::float4x4 makeZRotate( float angleRadians )
    {
        using simd::float4;
        const float a = angleRadians;
        return simd_matrix_from_rows((float4){ cosf( a ), sinf( a ), 0.0f, 0.0f },
                                     (float4){ -sinf( a ), cosf( a ), 0.0f, 0.0f },
                                     (float4){ 0.0f, 0.0f, 1.0f, 0.0f },
                                     (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
    }

    simd::float4x4 makeTranslate( const simd::float3& v )
    {
        using simd::float4;
        const float4 col0 = { 1.0f, 0.0f, 0.0f, 0.0f };
        const float4 col1 = { 0.0f, 1.0f, 0.0f, 0.0f };
        const float4 col2 = { 0.0f, 0.0f, 1.0f, 0.0f };
        const float4 col3 = { v.x, v.y, v.z, 1.0f };
        return simd_matrix( col0, col1, col2, col3 );
    }

    simd::float4x4 makeScale( const simd::float3& v )
    {
        using simd::float4;
        return simd_matrix((float4){ v.x,   0,   0,   0 },
                           (float4){   0, v.y,   0,   0 },
                           (float4){   0,   0, v.z,   0 },
                           (float4){   0,   0,   0, 1.0 });
    }

}


#pragma mark - Renderer
#pragma region Renderer {

const int Renderer::kMaxFramesInFlight = 3;

Renderer::Renderer( MTL::Device* pDevice )
: _pDevice( pDevice->retain() )
, _angle ( 0.f )
, _frame( 0 )
{
    _pCommandQueue = _pDevice->newCommandQueue();
    buildShaders();
    buildDepthStencilStates();
    buildBuffers();

    _semaphore = dispatch_semaphore_create( Renderer::kMaxFramesInFlight );
}

Renderer::~Renderer()
{
    _pShaderLibrary->release();
    _pDepthStencilState->release();
    _pVertexDataBuffer->release();
    for ( int i = 0; i < kMaxFramesInFlight; ++i )
    {
        _pInstanceDataBuffer[i]->release();
    }
    for ( int i = 0; i < kMaxFramesInFlight; ++i )
    {
        _pCameraDataBuffer[i]->release();
    }
    _pIndexBuffer->release();
    _pPSO->release();
    _pCommandQueue->release();
    _pDevice->release();
}

namespace shader_types
{
    struct VertexData {
        simd::float3 position;
    };

    struct InstanceData
    {
        simd::float4x4 instanceTransform;
        simd::float4 instanceColor;
    };

    struct CameraData
    {
        simd::float4x4 perspectiveTransform;
        simd::float4x4 worldTransform;
    };
}

void Renderer::buildShaders()
{
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newDefaultLibrary();
    if ( !pLibrary )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    MTL::Function* pVertexFn = pLibrary->newFunction( NS::String::string("vertexMain", UTF8StringEncoding) );
    MTL::Function* pFragFn = pLibrary->newFunction( NS::String::string("fragmentMain", UTF8StringEncoding) );

    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction( pVertexFn );
    pDesc->setFragmentFunction( pFragFn );
    pDesc->colorAttachments()->object(0)->setPixelFormat( MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );
    pDesc->setDepthAttachmentPixelFormat( MTL::PixelFormat::PixelFormatDepth16Unorm );

    _pPSO = _pDevice->newRenderPipelineState( pDesc, &pError );
    if ( !_pPSO )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    pVertexFn->release();
    pFragFn->release();
    pDesc->release();
    _pShaderLibrary = pLibrary;
}


// system used for depth comparison to ensure that the pixels drawn on the screen belong to objects that are the closest,
// not the objects that were drawn most recently.
void Renderer::buildDepthStencilStates()
{
    MTL::DepthStencilDescriptor* pDsDesc = MTL::DepthStencilDescriptor::alloc()->init(); // initialises depth comparison
    pDsDesc->setDepthCompareFunction( MTL::CompareFunction::CompareFunctionLess ); // sets the compare function to the one that looks
    // for the item closest to the camera
    pDsDesc->setDepthWriteEnabled( true ); // enables depth writing, so that when a closer color is written to a pixel, its new depth is written too.

    _pDepthStencilState = _pDevice->newDepthStencilState( pDsDesc );

    pDsDesc->release();
}

void Renderer::buildBuffers()
{
    using simd::float3;
    const float s = 0.5f;

//    const float ctp = cos( (2 * M_PI) / 3);
//    const float stp = sin( (2 * M_PI) / 3);
//    const float tSecond = s*stp;

    /// SIMPLE TETRAHEDRA
    float3 verts[] = { // tetrahedron made of equilateral triangles coordcs
        {+s, 0, 0},
        {-s / 3.0f, (2.0f * sqrt(2.0f)) * s / 3.0f, 0.0f},
        {-s / 3.0f, -sqrt(2.0f) * s / 3.0f, (sqrt(6.0f) / 3) * s},
        {-s / 3.0f, -sqrt(2.0f) * s / 3.0f, -(sqrt(6.0f) / 3) * s}
    };


//    float3 verts[] = { // 8 vertices of a triangle
//        { -s, -s, +s },
//        { +s, -s, +s },
//        { +s, +s, +s },
//        { -s, +s, +s },
//
//        { -s, -s, -s },
//        { -s, +s, -s },
//        { +s, +s, -s },
//        { +s, -s, -s }
//    };
//
//    uint16_t indices[] = { // all 12 triangles it takes to make 2 triangles for each of the 6 faces of a cube
//        0, 1, 2, /* front */
//        2, 3, 0,
//
//        1, 7, 6, /* right */
//        6, 2, 1,
//
//        7, 4, 5, /* back */
//        5, 6, 7,
//
//        4, 0, 3, /* left */
//        3, 5, 4,
//
//        3, 2, 6, /* top */
//        6, 5, 3,
//
//        4, 7, 1, /* bottom */
//        1, 0, 4
//    };




    uint16_t indices[] = { // all 12 triangles it takes to make 2 triangles for each of the 6 faces of a cube
        0, 1, 2, /* front */
        0, 2, 3,

        0, 3, 1, /* right */
        1, 3, 2,
    };

//    for(size_t i=0; i<lineWidth * kNumInstances; i++) {
//        directions[i] = simd::normalize(simd::float3{math::randomFloat(), math::randomFloat(), math::randomFloat()});
//    }


    const size_t vertexDataSize = sizeof( verts );
    const size_t indexDataSize = sizeof( indices );

    MTL::Buffer* pVertexBuffer = _pDevice->newBuffer( vertexDataSize, MTL::ResourceStorageModeManaged );
    MTL::Buffer* pIndexBuffer = _pDevice->newBuffer( indexDataSize, MTL::ResourceStorageModeManaged );

    _pVertexDataBuffer = pVertexBuffer;
    _pIndexBuffer = pIndexBuffer;

    memcpy( _pVertexDataBuffer->contents(), verts, vertexDataSize );
    memcpy( _pIndexBuffer->contents(), indices, indexDataSize );

    _pVertexDataBuffer->didModifyRange( NS::Range::Make( 0, _pVertexDataBuffer->length() ) );
    _pIndexBuffer->didModifyRange( NS::Range::Make( 0, _pIndexBuffer->length() ) );

    const size_t instanceDataSize = kMaxFramesInFlight * objCount * sizeof( shader_types::InstanceData );
    for ( size_t i = 0; i < kMaxFramesInFlight; ++i )
    {
        _pInstanceDataBuffer[ i ] = _pDevice->newBuffer( instanceDataSize, MTL::ResourceStorageModeManaged );
    }

    using simd::float3;
    using simd::float4;
    using simd::float4x4;

    const float scl = 0.5f;

    float3 objectPosition = { 0.f, 0.f, -10.f };

    size_t globalCtr = 0;

    for( size_t f=0; f<kMaxFramesInFlight; ++f) {

        shader_types::InstanceData* pInstanceData = reinterpret_cast< shader_types::InstanceData *>( _pInstanceDataBuffer[ f ]->contents() );
        globalCtr = 0;

        for( size_t j=0; j<lineWidth; j++) {
            for ( size_t i = 0; i < kNumInstances; ++i ) {
                float ictr = i / (float)kNumInstances;

                float xoff = (ictr * 4.0f - 2.0f) + (1.f/kNumInstances);
                float yoff = -1.5f;
                float zoff = (-1.0f + (2.0f*j+1)/lineWidth);


                float4x4 scale = math::makeScale( (float3) { scl, scl, scl } );
                float4x4 translate = math::makeTranslate( math::add( objectPosition, { xoff, yoff, zoff } ) );

                float4x4 pre = math::makeTranslate(objectPosition);
                float4x4 yRot = math::makeYRotate(0.01f * (f+1));
                float4x4 post = math::makeTranslate(float3{-objectPosition.x, -objectPosition.y, -objectPosition.z});

                pInstanceData[ globalCtr ].instanceTransform = translate * scale * (pre * yRot * post);

                float r = ictr;
                float g = 1.0f - r;
                float b = sinf( M_PI * ictr );
                pInstanceData[ globalCtr ].instanceColor = (float4){ r, g, b, 1.0f };

                // set direction of travel!!!
                directions[f][globalCtr] = simd::normalize(math::randomFloat3());
                globalCtr++;
            }
        }
        _pInstanceDataBuffer[f]->didModifyRange( NS::Range::Make( 0, _pInstanceDataBuffer[f]->length() ) );
    }


    const size_t cameraDataSize = kMaxFramesInFlight * sizeof( shader_types::CameraData );
    for ( size_t i = 0; i < kMaxFramesInFlight; ++i )
    {
        _pCameraDataBuffer[ i ] = _pDevice->newBuffer( cameraDataSize, MTL::ResourceStorageModeManaged );
    }
    _frame = 2;
}

void Renderer::draw( MTK::View* pView )
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;

    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();


//    _frame = (_frame + 1) % Renderer::kMaxFramesInFlight;
//    int lastFrame = _frame-1;
//    if(lastFrame < 0) lastFrame = kMaxFramesInFlight-1;
    int lastFrame = _frame = 1;

    MTL::Buffer* pInstanceDataBuffer = _pInstanceDataBuffer[ _frame ];
    MTL::Buffer* prevInstDataBuffer = _pInstanceDataBuffer[ lastFrame ];

    MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
    dispatch_semaphore_wait( _semaphore, DISPATCH_TIME_FOREVER );
    Renderer* pRenderer = this;
    pCmd->addCompletedHandler( ^void( MTL::CommandBuffer* pCmd ){
        dispatch_semaphore_signal( pRenderer->_semaphore );
    });

    _angle = 0.02f * kMaxFramesInFlight;

//    const float scl = 0.5f;
    shader_types::InstanceData* pInstanceData = reinterpret_cast< shader_types::InstanceData *>( pInstanceDataBuffer->contents() );
    shader_types::InstanceData* prevInstData = reinterpret_cast< shader_types::InstanceData *>( prevInstDataBuffer->contents() );


    for(int i=0; i<objCount; i++) {
        float3 lastPos = prevInstData[ i ].instanceTransform.columns[3].xyz;

//        std::cout << "lastFrame : " << lastFrame << ", i : " << i << ", _frame : " << _frame << "\n";
        float3 dir = directions[lastFrame][i];


//        float3 otherPos = prevInstData[ 1 ].instanceTransform.columns[3].xyz;
//        float3 toOtherVec = math::sub(otherPos, lastPos);
//
//        dir = math::add(dir, math::multByConstant(directions[lastFrame][1], directionConvergenceStrength) );
//        dir = math::add(dir, math::multByConstant(simd::normalize(toOtherVec), clusteringStrength) );
//        std::cout << "BEFORE direction vector : "<< dir.x << ", " << dir.y << ", " << dir.z << ". \n";
        for(int j=0; j<objCount; j++) {
            if(j == i) continue;
            float3 otherPos = prevInstData[ j ].instanceTransform.columns[3].xyz;

            float3 toOtherVec = math::sub(otherPos, lastPos);

            float dist = math::getDistanceSquared(lastPos, otherPos);

            if(dist > boidOuterRadSq) continue;

            else if( dist > boidInnerRadSq) {
                dir = math::add(dir, math::multByConstant(directions[lastFrame][j], directionConvergenceStrength) );
                dir = math::add(dir, math::multByConstant(toOtherVec, clusteringStrength) );
//                std::cout << "A";
            } else if(dist <= boidInnerRadSq){
                /// avoidance vector constant initially set to 3 to increase how much boids avoid collision at close proximity
                dir = math::add(dir, math::multByConstant(simd::float3{-toOtherVec.x, -toOtherVec.y, -toOtherVec.z}, avoidanceVectorStrength) );
//                std::cout << "B";
            }
        }

        float spd = abs(math::getMagnitude(dir));

        float r = math::fclamp(spd/200, 0, 1);
        float g = math::fclamp(spd/1000, 0, 1);
        float b = 0;

        pInstanceData[ i ].instanceColor = (float4){ r, g, b, 1.0f };

        directions[_frame][i] = simd::normalize(dir);
//        std::cout << "Position vector of boid " << i << " in frame " << lastFrame << " : " << lastPos.x << ", " << lastPos.y << ", " << lastPos.z << ". \n";

        pInstanceData[i].instanceTransform.columns[3].xyz = math::add(lastPos, math::multByConstant(dir, 0.003f));

        auto& posi = pInstanceData[i].instanceTransform.columns[3];
        if(posi.x > (windowSize/2)) posi.x -= windowSize;
        else if(posi.x < -(windowSize/2)) posi.x += windowSize;

        if(posi.y > (windowSize/2)) posi.y -= windowSize;
        else if(posi.y < -(windowSize/2)) posi.y += windowSize;

        if(posi.z > -(windowSize/2)) posi.z -= windowSize;
        else if(posi.z < -(3*windowSize/2)) posi.z += windowSize;
    }




//    int globalCtr = 0;
//    float3 objectPosition = {0, 0, 0.f};
//
//    for( size_t j=0; j<lineWidth; j++) {
//        for ( size_t i = 0; i < kNumInstances; ++i ) {
////            float3 objectPosition  = pInstanceData[ globalCtr ].instanceTransform.columns[3].xyz;
//
//            float4x4 rt = math::makeTranslate( objectPosition );
//            float4x4 rr = math::makeYRotate( _angle );
//            float4x4 rtInv = math::makeTranslate( { -objectPosition.x, -objectPosition.y, -objectPosition.z } );
//            float4x4 fullObjectRot = rt * rr * rtInv;
//
//            pInstanceData[ globalCtr ].instanceTransform *= fullObjectRot;
//
//            globalCtr++;
//
//        }
//    }
//
    pInstanceDataBuffer->didModifyRange( NS::Range::Make( 0, pInstanceDataBuffer->length() ) );

    // Update camera state:

    MTL::Buffer* pCameraDataBuffer = _pCameraDataBuffer[ _frame ];
    shader_types::CameraData* pCameraData = reinterpret_cast< shader_types::CameraData *>( pCameraDataBuffer->contents() );
    pCameraData->perspectiveTransform = math::makePerspective( 90.f * M_PI / 180.f, 1.f, 0.03f, 500.0f ) ;
    pCameraData->worldTransform = math::makeIdentity();
    pCameraDataBuffer->didModifyRange( NS::Range::Make( 0, sizeof( shader_types::CameraData ) ) );

    // Begin render pass:

    MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
    MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder( pRpd );

    pEnc->setRenderPipelineState( _pPSO );
    pEnc->setDepthStencilState( _pDepthStencilState );

    pEnc->setVertexBuffer( _pVertexDataBuffer, /* offset */ 0, /* index */ 0 );
    pEnc->setVertexBuffer( pInstanceDataBuffer, /* offset */ 0, /* index */ 1 );
    pEnc->setVertexBuffer( pCameraDataBuffer, /* offset */ 0, /* index */ 2 );

    pEnc->setCullMode( MTL::CullModeBack );
    pEnc->setFrontFacingWinding( MTL::Winding::WindingCounterClockwise );

//    std::cout << "GOT TO DRAWING!!! \n";

    pEnc->drawIndexedPrimitives( MTL::PrimitiveType::PrimitiveTypeTriangle,
                                12, MTL::IndexType::IndexTypeUInt16,
                                _pIndexBuffer,
                                0,
                                objCount );

    pEnc->endEncoding();
    pCmd->presentDrawable( pView->currentDrawable() );
    pCmd->commit();

    pPool->release();

//    std::this_thread::sleep_for(std::chrono::seconds(3));
}

#pragma endregion Renderer }
