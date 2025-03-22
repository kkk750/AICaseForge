<template>
  <div class="app-container">
    <nav :class="['navbar navbar-expand-lg', $route.path === '/' ? 'navbar-dark bg-transparent' : 'navbar-light bg-white shadow-sm']">
      <div class="container">
        <router-link class="navbar-brand" to="/">
          <img :src="$route.path === '/' ? '/static/images/logo-white.png' : '/static/images/logo-ai.png'" alt="Logo" height="40">
        </router-link>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav me-auto">
            <li class="nav-item">
              <router-link class="nav-link" to="/">主页</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/workbench">工作台</router-link>
            </li>
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">资源</a>
              <ul :class="['dropdown-menu', $route.path === '/' ? 'dropdown-menu-dark' : '']">
                <li><a class="dropdown-item" href="https://github.com/kkk750/AICaseForge" target="_blank">远程仓库</a></li>
                <li><a class="dropdown-item" href="#">Resource Link 2</a></li>
                <li><a class="dropdown-item" href="#">Resource Link 3</a></li>
              </ul>
            </li>
          </ul>
          <div class="navbar-nav">
            <a class="nav-link" href="https://github.com/kkk750/AICaseForge" target="_blank">Docs</a>
          </div>
        </div>
      </div>
    </nav>

    <main>
      <router-view></router-view>
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1030;
  transition: all 0.3s ease;
}

.navbar.bg-transparent {
  background-color: transparent !important;
}

.navbar.bg-white {
  background-color: #fff !important;
}

.shadow-sm {
  box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075) !important;
}
</style>
